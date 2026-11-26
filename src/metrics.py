"""Binary-classification metrics, bootstrap CIs, plots and small-cell suppression (PhysioNet rule of thumb:
never publish a count below 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

MIN_CELL = 10


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary(y, p, thr: float | None = None) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    out = dict(n=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()) if len(y) else np.nan)
    if 0 < y.sum() < len(y):
        out.update(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                   ece=ece(y, p))
    if thr is not None:
        pr = p >= thr
        out.update(threshold=thr, f1=f1_score(y, pr, zero_division=0), precision=precision_score(y, pr, zero_division=0),
                   recall=recall_score(y, pr, zero_division=0),
                   specificity=float(((~pr) & (y == 0)).sum() / max((y == 0).sum(), 1)))
    return out


def bootstrap_ci(y, p, fn=roc_auc_score, n: int = 200, seed: int = 0, max_n: int = 200_000):
    """Percentile 95% CI. Large sets are subsampled to `max_n` per resample (CI is then slightly conservative)."""
    y, p = np.asarray(y), np.asarray(p)
    if n <= 0 or y.sum() == 0 or y.sum() == len(y):
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    k = min(len(y), max_n)
    vals = []
    for _ in range(n):
        i = r.integers(0, len(y), k)
        if 0 < y[i].sum() < k:
            vals.append(fn(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def best_f1_threshold(y, p) -> float:
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return float(ts[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in ts]))])


def suppress(df: pd.DataFrame, count_cols=("n", "n_pos"), min_n: int = MIN_CELL) -> pd.DataFrame:
    """Drop rows whose counts could identify few patients (n < 10). Zero positives is not a small cell."""
    df = df.copy()
    keep = np.ones(len(df), bool)
    for c in count_cols:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
            keep &= (v == 0) | (v >= min_n)
    return df[keep].reset_index(drop=True)


def plot_roc_pr(curves: dict, path, pos_label="positive"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, (y, p) in curves.items():
        y, p = np.asarray(y), np.asarray(p)
        if not (0 < y.sum() < len(y)):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        k = max(1, len(fpr) // 2000)
        ax[0].plot(fpr[::k], tpr[::k], label=f"{name} ({roc_auc_score(y, p):.3f})")
        k = max(1, len(rc) // 2000)
        ax[1].plot(rc[::k], pr[::k], label=f"{name} ({average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (AUROC)")
    ax[1].set(xlabel=f"Recall ({pos_label})", ylabel=f"Precision ({pos_label})", title="Precision-recall (AUPRC)")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(curves: dict, path, bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for name, (y, p) in curves.items():
        y, p = np.asarray(y, float), np.asarray(p, float)
        q = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= MIN_CELL]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= MIN_CELL]
        ax.plot(xs, ys, "o-", ms=3, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set(xlabel="Predicted probability", ylabel="Observed rate", title="Calibration (deciles)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    if df is None or not len(df):
        return "–"

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])
