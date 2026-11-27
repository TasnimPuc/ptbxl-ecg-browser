# ECG in your Browser — PTB-XL 1D-CNN → ONNX 🫀

![repo health](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/TasnimPuc/bd-rooftop-solar-calculator/main/.github/health/badge.json)

A small 1D-CNN reads a **12-lead ECG** and estimates five diagnostic superclasses: **NORM, MI, STTC, CD, HYP**.
It runs **entirely in the browser** on GitHub Pages with ONNX Runtime Web. There is no server, and no ECG ever
leaves the visitor's computer.

**Live demo:** `https://<your-username>.github.io/ptbxl-ecg-browser/` (after the setup below)

## What is inside
- **Data:** all 21,799 records of [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), 12 leads, 10 s at
  100 Hz. Labels are the diagnostic superclasses. Records without any superclass are dropped, as in Strodthoff et
  al. (2020).
- **Official split:** folds 1–8 train, fold 9 validation, fold 10 test (human-validated labels).
- **Model:** a residual CNN with depthwise-separable convolutions, about 0.1 M parameters and well under 1 MB as ONNX.
  Per-lead normalisation is built into the model, so the browser sends raw millivolts.
- **Training:** AdamW, cosine schedule, augmentation (amplitude, baseline, noise, time shift), early stopping on
  validation macro AUROC.
- **Evaluation:** test-fold macro AUROC with a bootstrap 95% CI, plus per-class AUROC, AUPRC, F1, sensitivity and
  specificity. Each class has its own threshold, chosen on fold 9.
- **Export:** ONNX (opset 17, sigmoid inside the graph), checked against PyTorch with onnxruntime, plus CPU latency.

## Run on Kaggle (~15–30 min on one T4)
1. Import `notebooks/kaggle_run.ipynb`. Set Accelerator **GPU T4** and Internet **On**.
2. **Data:** nothing to do. An attached PTB-XL copy is used if there is one; otherwise the notebook downloads it from
   PhysioNet once (open access).
3. *(recommended)* Add-ons → **Google Cloud Services** → tick **Cloud Storage**, and set `project_id`. The prepared
   `.npz` and every checkpoint then live in your bucket (`gs://<project_id>-mimic-ckpt/ptbxl-ecg-browser/`). If a
   session stops, Run All continues **mid-epoch** and skips the download.
4. **Run All**, then download `results_for_github.zip` and unzip it at the repo root. This adds `docs/model.onnx`,
   `docs/meta.json`, `docs/samples.json`, `models/ecgnet.pth`, `models/ecgnet.onnx` and `results/`.

## Publish the demo (GitHub Pages)
Repo → **Settings → Pages → Build and deployment: Deploy from a branch → `main` / `/docs`** → Save.
After about a minute the site is live. Visitors can:
- pick one of the PTB-XL test ECGs shipped with the demo (3 per class, CC BY 4.0), or
- upload their own CSV: 12 columns (I, II, III, aVR, aVL, aVF, V1–V6), one row per sample, 100–1000 Hz, in mV or µV.
  The demo resamples it and uses the central 10 s.

## Results
See **[results/REPORT.md](results/REPORT.md)** after the run. For reference, Strodthoff et al. (2020) report a macro
AUROC of about 0.93 on this split with much larger models. This model gives up a little accuracy to be small and fast.

## Local tests / CI
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
npm install --prefix /tmp/ortnode onnxruntime-web@1.29.0       # optional: test the browser runtime in Node
python -m pytest -q            # fake PTB-XL (real wfdb files) -> train + resume -> ONNX parity ->
                               # docs/ecg.js + onnxruntime-web gives the same probabilities as Python
python tools/build_notebook.py # regenerate the Kaggle notebook after editing src/
```
To preview the site locally, run `python -m http.server -d docs 8000` and open `http://localhost:8000`.

## Licence and citation
- Code: MIT. PTB-XL: CC BY 4.0. Please cite Wagner P. et al., *PTB-XL, a large publicly available
  electrocardiography dataset*, Scientific Data 7, 154 (2020), and the PhysioNet record (doi:10.13026/kfzx-aw45).
- **Research demo. This is not a medical device and must not be used for diagnosis.**
