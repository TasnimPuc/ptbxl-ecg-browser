from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- data: PTB-XL v1.0.3 (open access, CC BY 4.0), all 21,799 records, 100 Hz ---
    data_dir: str = ""                    # "" = search /kaggle/input for ptbxl_database.csv, else download from PhysioNet
    download_url: str = "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
    sampling_rate: int = 100              # 100 Hz x 10 s = 1000 samples per lead (small enough for the browser)
    task: str = "superdiagnostic"         # 5 classes: NORM, MI, STTC, CD, HYP (multi-label)
    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8)   # official split: 9 = validation, 10 = test
    val_fold: int = 9
    test_fold: int = 10
    seed: int = 42

    # --- compact 1D-CNN ---
    widths: tuple = (32, 64, 96, 128)
    kernel: int = 7
    dropout: float = 0.2
    batch_size: int = 128
    lr: float = 3e-3
    epochs: int = 60
    patience: int = 10
    augment: bool = True                  # amplitude scaling, baseline shift, noise, time shift

    # --- evaluation / export ---
    n_boot: int = 500
    demo_per_class: int = 3               # example ECGs (test fold) shipped with the web demo
    opset: int = 17

    # --- outputs + checkpoints ---
    work_dir: str = "/kaggle/working/ecg"
    cache_dir: str = "/tmp/ecg_cache"
    project_id: str = ""                  # GCS bucket for data cache + checkpoints ("" = local only, no resume across sessions)
    bucket: str = ""
    bucket_location: str = "US"
    gcs_prefix: str = "ptbxl-ecg-browser"
    run_name: str = "main"
    local_store: str = ""
    ckpt_every_min: float = 5.0
    device: str = ""

    def to_dict(self):
        return asdict(self)
