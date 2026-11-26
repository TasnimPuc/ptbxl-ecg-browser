from __future__ import annotations

import shutil
from pathlib import Path

from .metrics import md


def build(out: Path, pc, split_tab, s: dict) -> str:
    res = out / "results"
    (res / "figures").mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy(f, res / "figures" / f.name)
    pc.to_csv(res / "per_class.csv", index=False)
    lo, hi = s["macro_auroc_ci"]
    o = s["onnx"]
    text = f"""# Results — PTB-XL 5-class ECG classifier (browser-sized)

**Data:** PTB-XL v1.0.3, all {s['records']:,} records, 12 leads, 10 s at {s['config']['sampling_rate']} Hz.
Official split: folds 1–8 train, 9 validation, 10 test (human-validated). Task: diagnostic superclasses
(NORM, MI, STTC, CD, HYP), multi-label.

**Model:** compact residual 1D-CNN with depthwise-separable convolutions, **{s['params']:,} parameters**,
ONNX **{o['onnx_kb']} KB**, {o['cpu_ms_per_ecg']} ms per ECG on one CPU core (onnxruntime). ONNX vs PyTorch max
difference: {o['max_abs_diff']:.2e}.

## Test fold
**Macro AUROC {s['macro_auroc']:.4f}** (95% bootstrap CI {lo:.4f}–{hi:.4f}) · best epoch {s['best_epoch']}.

{md(pc.drop(columns=['n']))}

![ROC](figures/roc_test.png)

## Data
{md(__import__('pandas').DataFrame(s['split']))}

## Example (test fold)
![example](figures/example_ecg.png)

---
Runtime {s['runtime_min']} min. Reference point: Strodthoff et al. (2020) report macro AUROC ≈ 0.93 for much
larger models (e.g. xresnet1d101) on the same split; this model trades a little accuracy for being small enough to
run in a web page. Not a medical device.
"""
    (res / "REPORT.md").write_text(text, encoding="utf-8")
    return text
