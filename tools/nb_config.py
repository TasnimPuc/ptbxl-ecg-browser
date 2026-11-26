"""Project-specific texts for tools/build_notebook.py."""
ACCELERATOR = "nvidiaTeslaT4"
EXTRA_PIP = "wfdb onnx onnxruntime"

INTRO = """# 🫀 PTB-XL ECG Classifier → ONNX → Browser (Kaggle)

Trains a **small 1D-CNN** (~0.1 M parameters) on **all 21,799 PTB-XL ECGs** to recognise five diagnostic superclasses,
exports it to **ONNX**, and writes a ready-to-publish **GitHub Pages** site (`docs/`) where the model runs entirely
in the visitor's browser. No server is needed.

**Before Run All**
1. Accelerator **GPU T4** · Internet **On**.
2. Data: nothing to do. If a PTB-XL copy is attached under *Add Input* it is used; otherwise it downloads from
   PhysioNet (open access, 1.7 GB) once.
3. Optional but recommended: Add-ons → **Google Cloud Services** → tick **Cloud Storage**, and set `project_id`. Then
   the prepared data (.npz) and all checkpoints live in your bucket, and a stopped session resumes mid-epoch.

At the end, download **`results_for_github.zip`** and unzip it at the repo root. It contains `docs/` (the web demo),
`models/` (`.pth` + `.onnx`) and `results/` (report and figures). PTB-XL is CC BY 4.0, so all of it may be published."""

CONFIG = """import sys
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

cfg = Config(
    project_id="mimic-cxr-training",   # <-- GCP project for the checkpoint bucket ("" = no bucket, no cross-session resume)
    work_dir="/kaggle/working/ecg",
    epochs=60, batch_size=128, lr=3e-3,
    widths=(32, 64, 96, 128),          # model size (bigger = more accurate, heavier web page)
    run_name="main",                   # change to start a fresh set of checkpoints
)
cfg"""

CHECK_MD = "## 4 · Pre-flight check\nGPU, and the Cloud Storage bucket (only if `project_id` is set)."
CHECK_CODE = """import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU, slower)")
if cfg.project_id:
    from src.store import make_store
    s = make_store(cfg)
    print("bucket OK:", s.uri())
else:
    print("no project_id: checkpoints stay in /kaggle/working only")"""

SHOW_NOTE = "Everything here may be published (open data): web demo, model files, report."
SHOW_CODE = """import re, shutil, glob, os
from IPython.display import Markdown, Image, display
w = cfg.work_dir
text = open(f"{w}/results/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", text)))
for f in sorted(glob.glob(f"{w}/results/figures/*.png")):
    display(Image(f))
pkg = "/kaggle/working/github_package"
shutil.rmtree(pkg, ignore_errors=True)
for sub in ("results", "models"):
    shutil.copytree(f"{w}/{sub}", f"{pkg}/{sub}")
os.makedirs(f"{pkg}/docs", exist_ok=True)
for f in ("model.onnx", "meta.json", "samples.json"):
    shutil.copy(f"{w}/docs/{f}", f"{pkg}/docs/{f}")
shutil.make_archive("/kaggle/working/results_for_github", "zip", pkg)
print("Download /kaggle/working/results_for_github.zip and unzip it at the repo root (docs/, models/, results/).")"""
