from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .data import CLASS_NAMES, CLASSES
from .metrics import best_f1_threshold


def thresholds(Yv, Pv) -> np.ndarray:
    """Per-class decision threshold = best F1 on the validation fold."""
    return np.array([best_f1_threshold(Yv[:, j], Pv[:, j]) for j in range(Yv.shape[1])])


def per_class(Y, P, thr) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(CLASSES):
        y, p = Y[:, j], P[:, j]
        rows.append(dict(cls=c, name=CLASS_NAMES[c], n=int(len(y)), positives=int(y.sum()), auroc=roc_auc_score(y, p),
                         auprc=average_precision_score(y, p), threshold=float(thr[j]),
                         f1=f1_score(y, p >= thr[j], zero_division=0),
                         sensitivity=float(((p >= thr[j]) & (y == 1)).sum() / max(y.sum(), 1)),
                         specificity=float(((p < thr[j]) & (y == 0)).sum() / max((y == 0).sum(), 1))))
    return pd.DataFrame(rows)


def macro_ci(Y, P, n=500, seed=0):
    r = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = r.integers(0, len(Y), len(Y))
        a = [roc_auc_score(Y[i, j], P[i, j]) for j in range(Y.shape[1]) if 0 < Y[i, j].sum() < len(i)]
        if a:
            vals.append(np.mean(a))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def plot_roc(Y, P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for j, c in enumerate(CLASSES):
        f, t, _ = roc_curve(Y[:, j], P[:, j])
        ax.plot(f, t, label=f"{c} ({roc_auc_score(Y[:, j], P[:, j]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.7)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="PTB-XL test fold — ROC per class")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(x, leads, probs, thr, true, path, fs=100):
    """One test ECG (public PTB-XL) with the model's output — figure for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(x.shape[0]) / fs
    fig, axes = plt.subplots(12, 1, figsize=(9, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x[:, i], lw=.8, c="k")
        ax.set_ylabel(leads[i], rotation=0, labelpad=14, fontsize=7)
        ax.set_yticks([])
        ax.grid(alpha=.25)
    axes[-1].set_xlabel("seconds")
    txt = "  ".join(f"{c}: {p:.2f}{' ✓' if p >= th else ''}" for c, p, th in zip(CLASSES, probs, thr))
    fig.suptitle(f"model: {txt}\ntrue: {', '.join(c for c, y in zip(CLASSES, true) if y)}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
