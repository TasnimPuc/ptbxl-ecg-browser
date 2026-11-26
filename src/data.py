"""PTB-XL loading: find the dataset (Kaggle input or PhysioNet download), read all records with wfdb once, and cache
one compact .npz (float16) in the bucket so later sessions skip the 1.7 GB download and the 21,799 file reads."""
from __future__ import annotations

import ast
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "STTC": "ST/T change",
               "CD": "Conduction disturbance", "HYP": "Hypertrophy"}


def locate(cfg) -> Path:
    if cfg.data_dir and (Path(cfg.data_dir) / "ptbxl_database.csv").exists():
        return Path(cfg.data_dir)
    for root in (Path("/kaggle/input"), Path(cfg.cache_dir)):
        if root.exists():
            hits = sorted(root.rglob("ptbxl_database.csv"))
            if hits:
                return hits[0].parent
    return download(cfg)


def download(cfg) -> Path:
    """PhysioNet zip (1.7 GB). Only the CSVs and the records of the chosen sampling rate are extracted."""
    import requests
    out = Path(cfg.cache_dir) / "ptbxl"
    z = Path(cfg.cache_dir) / "ptbxl.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    if not z.exists():
        log.info("downloading PTB-XL from %s (1.7 GB) ...", cfg.download_url)
        with requests.get(cfg.download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = z.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            tmp.replace(z)
    folder = "records100/" if cfg.sampling_rate == 100 else "records500/"
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith(("ptbxl_database.csv", "scp_statements.csv")) or folder in n]
        zf.extractall(out, members=names)
    return next(out.rglob("ptbxl_database.csv")).parent


def labels(db: pd.DataFrame, scp: pd.DataFrame) -> np.ndarray:
    diag = scp[scp.diagnostic == 1]
    Y = np.zeros((len(db), len(CLASSES)), np.uint8)
    for i, codes in enumerate(db.scp_codes):
        for code in ast.literal_eval(codes) if isinstance(codes, str) else codes:
            if code in diag.index:
                c = diag.loc[code, "diagnostic_class"]
                if c in CLASSES:
                    Y[i, CLASSES.index(c)] = 1
    return Y


def build(root: Path, cfg) -> dict:
    import wfdb
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    col = "filename_lr" if cfg.sampling_rate == 100 else "filename_hr"
    files = [str(root / f) for f in db[col]]

    def rd(f):
        sig, meta = wfdb.rdsamp(f)
        return sig.astype(np.float16), meta["sig_name"]
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(rd, files))
    X = np.stack([r[0] for r in res])
    return dict(X=X, Y=labels(db, scp), fold=db.strat_fold.to_numpy(np.int8), ecg_id=db.index.to_numpy(np.int32),
                leads=np.array(res[0][1]), classes=np.array(CLASSES))


def load(cfg, store) -> dict:
    key = f"data/ptbxl-{cfg.sampling_rate}hz-v1.0.3.npz"
    local = Path(cfg.cache_dir) / Path(key).name
    if not local.exists() and store.exists(key):
        log.info("PTB-XL cache found in bucket -> download skipped")
        store.get(key, local)
    if local.exists():
        d = dict(np.load(local, allow_pickle=False))
    else:
        root = locate(cfg)
        log.info("reading all PTB-XL records from %s (once)", root)
        d = build(root, cfg)
        local.parent.mkdir(parents=True, exist_ok=True)
        np.savez(local, **d)
        store.put(local, key)
    log.info("PTB-XL: %d records, X %s, labelled %d", len(d["Y"]), d["X"].shape, int((d["Y"].sum(1) > 0).sum()))
    return d


def splits(d: dict, cfg) -> dict:
    """Official folds; records without any superclass label are dropped (as in Strodthoff et al., 2020)."""
    ok = d["Y"].sum(1) > 0
    f = d["fold"]
    return {"train": np.flatnonzero(ok & np.isin(f, cfg.train_folds)), "val": np.flatnonzero(ok & (f == cfg.val_fold)),
            "test": np.flatnonzero(ok & (f == cfg.test_fold))}
