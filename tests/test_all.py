"""Synthetic PTB-XL-shaped folder (real wfdb files + CSVs) -> full pipeline (train with resume, ONNX parity, demo
assets) -> the web demo's JavaScript (docs/ecg.js + onnxruntime-web in Node, if available) gives the same output."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import Config

ROOT = Path(__file__).resolve().parents[1]
CODES = {"NORM": "NORM", "IMI": "MI", "NDT": "STTC", "CLBBB": "CD", "LVH": "HYP"}


def _fake_ptbxl(root: Path, n=320, seed=0):
    import wfdb
    r = np.random.default_rng(seed)
    (root / "records100" / "00000").mkdir(parents=True)
    rows = []
    t = np.arange(1000) / 100
    for i in range(1, n + 1):
        code = list(CODES)[i % 5]
        beat = np.sin(2 * np.pi * 1.2 * t) ** 21                     # spiky "QRS" train
        sig = np.stack([beat * (1 + 0.1 * l) for l in range(12)], 1) * 0.8 + r.normal(0, .05, (1000, 12))
        k = list(CODES).index(code)
        if k:
            sig[:, k + 5] += (0.6 if k % 2 else -0.6) * np.sin(2 * np.pi * 1.2 * t + 1)   # class-specific lead change
        name = f"{i:05d}_lr"
        wfdb.wrsamp(name, fs=100, units=["mV"] * 12, sig_name=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4",
                                                                "V5", "V6"], p_signal=sig, fmt=["16"] * 12,
                    write_dir=str(root / "records100" / "00000"))
        rows.append(dict(ecg_id=i, patient_id=i, scp_codes=str({code: 100.0, "SR": 0.0}), strat_fold=((i // 5) % 10) + 1,
                         filename_lr=f"records100/00000/{name}", filename_hr=f"records500/00000/{i:05d}_hr"))
    rows.append(dict(ecg_id=n + 1, patient_id=n + 1, scp_codes=str({"SR": 0.0}), strat_fold=1,    # no superclass
                     filename_lr=f"records100/00000/{1:05d}_lr", filename_hr=""))
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    scp = pd.DataFrame({"diagnostic": [1, 1, 1, 1, 1, np.nan], "diagnostic_class": list(CODES.values()) + [np.nan]},
                       index=list(CODES) + ["SR"])
    scp.to_csv(root / "scp_statements.csv")
    return root


def _cfg(tmp, work="o", **kw):
    base = dict(data_dir=str(tmp / "ptbxl"), work_dir=str(tmp / work), cache_dir=str(tmp / f"cache_{work}"),
                local_store=str(tmp / "bucket"), epochs=6, batch_size=32, n_boot=20, demo_per_class=1,
                widths=(16, 24), device="cpu", lr=5e-3)
    base.update(kw)
    return Config(**base)


@pytest.fixture(scope="module")
def ptb(tmp_path_factory):
    return _fake_ptbxl(tmp_path_factory.mktemp("d") / "ptbxl").parent


def test_load_labels_and_splits(ptb):
    from src.store import LocalStore
    cfg = _cfg(ptb, "load")
    d = data.load(cfg, LocalStore(cfg.local_store, "x"))
    assert d["X"].shape == (321, 1000, 12) and d["X"].dtype == np.float16
    assert list(d["leads"])[:2] == ["I", "II"] and d["Y"][-1].sum() == 0
    sp = data.splits(d, cfg)
    assert 321 - 1 == sum(len(v) for v in sp.values())                 # unlabelled record dropped
    assert set(d["fold"][sp["test"]]) == {10} and set(d["fold"][sp["val"]]) == {9}
    d2 = data.load(_cfg(ptb, "load2"), LocalStore(cfg.local_store, "x"))   # second session: from the bucket cache
    assert np.array_equal(d["X"], d2["X"])


def test_pipeline_onnx_demo_and_resume(ptb):
    from src.pipeline import run
    cfg = _cfg(ptb)
    s = run(cfg)
    assert s["macro_auroc"] > 0.8 and s["onnx"]["max_abs_diff"] < 1e-4
    docs = Path(cfg.work_dir) / "docs"
    meta = json.loads((docs / "meta.json").read_text())
    samples = json.loads((docs / "samples.json").read_text())
    assert meta["n_samples"] == 1000 and len(meta["thresholds"]) == 5 and len(samples) >= 3
    assert len(samples[0]["signal"]) == 12 and len(samples[0]["signal"][0]) == 1000
    assert (Path(cfg.work_dir) / "models" / "ecgnet.pth").exists()
    s2 = run(_cfg(ptb, "o2"))                                            # new session: training restored
    assert abs(s2["macro_auroc"] - s["macro_auroc"]) < 1e-9


def test_web_demo_javascript_matches_python(ptb):
    """docs/ecg.js preprocessing + onnxruntime-web (the browser runtime) in Node == Python onnxruntime."""
    node = shutil.which("node")
    ortweb = Path("/tmp/ortnode/node_modules/onnxruntime-web")
    if not node or not ortweb.exists():
        pytest.skip("node / onnxruntime-web not installed")
    import onnxruntime as ort
    from src.pipeline import run
    cfg = _cfg(ptb, "js", epochs=1)
    run(cfg)
    docs = Path(cfg.work_dir) / "docs"
    sample = json.loads((docs / "samples.json").read_text())[0]
    x = np.array(sample["signal"], np.float32)[None]
    ref = ort.InferenceSession(str(docs / "model.onnx")).run(None, {"ecg": x})[0][0]
    js = f"""
const ort = require("{ortweb}");
const ECG = require("{ROOT / 'docs' / 'ecg.js'}");
const fs = require("fs");
(async () => {{
  const s = JSON.parse(fs.readFileSync("{docs / 'samples.json'}"))[0];
  const leads = s.signal.map((l) => Float32Array.from(l));
  const x = ECG.toModelInput(leads, 100, 100, 1000, ECG.guessScale(leads));
  const sess = await ort.InferenceSession.create(fs.readFileSync("{docs / 'model.onnx'}"));
  const out = await sess.run({{ ecg: new ort.Tensor("float32", x, [1, 12, 1000]) }});
  console.log(JSON.stringify(Array.from(out.probabilities.data)));
  // 500 Hz upload path: upsample x5 and back -> same length
  const up = leads.map((l) => ECG.resample(l, 100, 500));
  const x2 = ECG.toModelInput(up, 500, 100, 1000, 1);
  console.log(x2.length);
}})();
"""
    f = Path(cfg.work_dir) / "t.js"
    f.write_text(js)
    out = subprocess.run([node, str(f)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    got = np.array(json.loads(lines[0]))
    assert np.abs(got - ref).max() < 1e-4 and lines[1] == "12000"
    csv = "I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6\n" + "\n".join(",".join("0.1" for _ in range(12)) for _ in range(5))
    chk = subprocess.run([node, "-e", f"const E=require('{ROOT / 'docs' / 'ecg.js'}');const l=E.parseCSV(`{csv}`);"
                                      "console.log(l.length, l[0].length)"], capture_output=True, text=True)
    assert chk.stdout.strip() == "12 5"


def test_notebook_steps(ptb):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from nbsteps import exec_steps
    ns = exec_steps(ROOT / "src" / "pipeline.py", _cfg(ptb, "nb", epochs=1))
    assert ns["result"]["records"] == 321
