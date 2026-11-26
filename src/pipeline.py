"""PTB-XL (all records) -> compact 1D-CNN (resumable) -> test-fold evaluation -> ONNX (+ parity check) ->
GitHub Pages demo assets (docs/) -> report. Open data: weights, ONNX and examples may be published."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, evaluate, export
from .config import Config
from .metrics import md
from .model import n_params
from .store import LocalStore
from .train import macro_auc, predict, train
from .trainer import history

log = logging.getLogger("ecg")


def run(cfg: Config) -> dict:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    for noisy in ("google", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(cfg.work_dir)
    for sub in ("figures", "models", "results", "docs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if cfg.project_id or cfg.local_store:
        from .store import make_store
        store = make_store(cfg)
    else:                                   # no bucket: works, but a new Kaggle session cannot resume
        store = LocalStore(out / "local_bucket", cfg.gcs_prefix)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("device: %s | checkpoints: %s", device, store.uri())

    # %% Load PTB-XL | All 21,799 records at 100 Hz. First run: Kaggle input or PhysioNet download (1.7 GB), read once with wfdb and cached as one .npz in your bucket; later runs skip all of that.
    # show: split_tab
    d = data.load(cfg, store)
    sp = data.splits(d, cfg)
    split_tab = pd.DataFrame([{"part": k, "records": len(v), **{c: int(d["Y"][v, j].sum()) for j, c in enumerate(data.CLASSES)}}
                              for k, v in sp.items()])

    # %% Train the 1D-CNN (resumable mid-epoch) | Official folds 1–8 train, 9 validation. Model, optimizer, scheduler and the position inside the epoch are saved every few minutes.
    # show: history(st, "val_macro_auroc")
    fp = hashlib.sha1(json.dumps([len(d["Y"]), list(cfg.widths), cfg.kernel, cfg.dropout, cfg.lr, cfg.batch_size,
                                  cfg.augment, cfg.seed]).encode()).hexdigest()[:8]
    model, st = train(cfg, d, sp, device, store, f"ckpt/{cfg.run_name}-{cfg.sampling_rate}hz-{fp}")
    log.info("parameters: %s", f"{n_params(model):,}")

    # %% Evaluate on the test fold | Fold 10 (human-validated labels). Per-class thresholds = best F1 on fold 9.
    # show: pc
    Pv, Pt = predict(model, d["X"], sp["val"], device), predict(model, d["X"], sp["test"], device)
    Yv, Yt = d["Y"][sp["val"]], d["Y"][sp["test"]]
    thr = evaluate.thresholds(Yv, Pv)
    pc = evaluate.per_class(Yt, Pt, thr)
    ci = evaluate.macro_ci(Yt, Pt, cfg.n_boot, cfg.seed)
    evaluate.plot_roc(Yt, Pt, out / "figures" / "roc_test.png")
    i0 = int(np.argmax(Pt[:, 1] * Yt[:, 1])) if Yt[:, 1].any() else 0
    evaluate.plot_example(d["X"][sp["test"][i0]].astype(np.float32), d["leads"], Pt[i0], thr, Yt[i0],
                          out / "figures" / "example_ecg.png", cfg.sampling_rate)

    # %% Export to ONNX | Probabilities = sigmoid inside the graph; per-lead normalisation is inside the model, so the browser sends raw mV. Checked against onnxruntime.
    # show: pd.Series(onnx_info)
    onnx_path = export.to_onnx(model, out / "docs" / "model.onnx", d["X"].shape[2], d["X"].shape[1], cfg.opset)
    onnx_info = export.check(model, onnx_path, d["X"][sp["test"][:64]])
    model.to(device)

    # %% Web demo assets + report | `docs/` = GitHub Pages site (model.onnx, meta.json, samples.json). `.pth` is included: PTB-XL is open data (CC BY 4.0).
    # show: pd.Series({"demo examples": n_demo, "macro AUROC (test)": round(summary["macro_auroc"], 4)})
    summary = dict(records=len(d["Y"]), split=split_tab.to_dict("records"), params=n_params(model),
                   macro_auroc=macro_auc(Yt, Pt), macro_auroc_ci=list(ci), best_epoch=st.best_epoch,
                   per_class=pc.to_dict("records"), onnx=onnx_info, runtime_min=round((time.time() - t0) / 60, 1),
                   config=cfg.to_dict())
    n_demo = export.demo_assets(out / "docs", d, sp["test"], Pt, thr, summary, pc.to_dict("records"), cfg)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "classes": data.CLASSES,
                "leads": [str(x) for x in d["leads"]], "fs": cfg.sampling_rate, "widths": list(cfg.widths),
                "kernel": cfg.kernel, "dropout": cfg.dropout, "thresholds": thr.tolist()}, out / "models" / "ecgnet.pth")
    shutil.copy(onnx_path, out / "models" / "ecgnet.onnx")
    (out / "results" / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    from .report import build as build_report
    build_report(out, pc, split_tab, summary)
    store.put_dir(out / "results", f"runs/{cfg.run_name}/results")
    store.put_dir(out / "models", f"runs/{cfg.run_name}/models")
    log.info("DONE in %.1f min | test macro AUROC %.4f", summary["runtime_min"], summary["macro_auroc"])
    return summary
