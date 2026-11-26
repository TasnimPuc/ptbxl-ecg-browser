"""ONNX export (+ onnxruntime parity check and CPU latency) and the files the GitHub Pages demo needs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import CLASS_NAMES, CLASSES
from .model import Probabilities


def to_onnx(model, path: Path, n_leads: int, n_samples: int, opset: int = 17) -> Path:
    m = Probabilities(model).cpu().eval()
    dummy = torch.zeros(1, n_leads, n_samples)
    kw = dict(input_names=["ecg"], output_names=["probabilities"], opset_version=opset,
              dynamic_axes={"ecg": {0: "batch"}, "probabilities": {0: "batch"}})
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)     # classic exporter: no onnxscript needed
    try:
        torch.onnx.export(m, dummy, str(path), dynamo=False, **kw)
    except TypeError:                                          # older torch: no `dynamo` argument
        torch.onnx.export(m, dummy, str(path), **kw)
    return path


def check(model, path: Path, X: np.ndarray) -> dict:
    """Max |torch - onnxruntime| on real ECGs and single-ECG CPU latency."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = X.astype(np.float32).transpose(0, 2, 1)
    with torch.no_grad():
        ref = Probabilities(model).cpu().eval()(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"ecg": x})[0]
    one = x[:1]
    sess.run(None, {"ecg": one})
    t = time.time()
    for _ in range(20):
        sess.run(None, {"ecg": one})
    return {"max_abs_diff": float(np.abs(ref - got).max()), "cpu_ms_per_ecg": round((time.time() - t) / 20 * 1000, 2),
            "onnx_kb": round(path.stat().st_size / 1024, 1)}


def demo_assets(out: Path, d, test_idx, P_test, thr, summary: dict, per_class_rows, cfg):
    """docs/meta.json + docs/samples.json: a few PTB-XL test ECGs (CC BY 4.0) to try in the browser."""
    out.mkdir(parents=True, exist_ok=True)
    Y = d["Y"]
    picked = []
    for j in range(len(CLASSES)):                               # confident, correct examples of each class
        cand = [i for i in np.argsort(-P_test[:, j]) if Y[test_idx[i], j] == 1 and P_test[i, j] >= thr[j]]
        picked += [i for i in cand if i not in picked][: cfg.demo_per_class]
    samples = []
    for i in picked:
        r = test_idx[i]
        samples.append({"ecg_id": int(d["ecg_id"][r]), "true": [c for c, y in zip(CLASSES, Y[r]) if y],
                        "signal": np.round(d["X"][r].astype(np.float32).T, 3).tolist()})   # [12][T] in mV
    meta = {"classes": CLASSES, "names": [CLASS_NAMES[c] for c in CLASSES], "thresholds": [round(float(t), 4) for t in thr],
            "leads": [str(x) for x in d["leads"]], "fs": cfg.sampling_rate, "n_samples": int(d["X"].shape[1]),
            "test": {"macro_auroc": summary["macro_auroc"], "macro_auroc_ci": summary["macro_auroc_ci"],
                     "per_class": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                   for r in per_class_rows]},
            "model": {"params": summary["params"], "onnx_kb": summary["onnx"]["onnx_kb"]},
            "source": "PTB-XL v1.0.3 (Wagner et al., 2020/2022), CC BY 4.0"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "samples.json").write_text(json.dumps(samples, separators=(",", ":")))
    return len(samples)
