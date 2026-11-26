"""Training with the resumable trainer (mid-epoch checkpoints in the bucket) + GPU augmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .model import ECGNet
from .trainer import Checkpointer, TensorBatches, cosine_warmup, fit


def augment(x):
    """x [B, 12, T] in mV: amplitude scaling, baseline offset, gaussian noise, circular time shift."""
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 1, 1, device=x.device))
    x = x + 0.05 * torch.randn(B, x.shape[1], 1, device=x.device)
    x = x + 0.01 * torch.randn_like(x)
    shift = int(torch.randint(-50, 51, (1,)).item())
    return torch.roll(x, shift, dims=-1)


@torch.no_grad()
def predict(model, X, idx, device, bs=1024) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(X[idx[s:s + bs]].astype(np.float32)).to(device).transpose(1, 2)
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 5))


def macro_auc(Y, P):
    a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(a)) if a else float("nan")


def train(cfg, d, sp, device, store, key):
    X, Y = d["X"], d["Y"].astype(np.float32)
    tr, va = sp["train"], sp["val"]
    torch.manual_seed(cfg.seed)
    model = ECGNet(X.shape[2], Y.shape[1], cfg.widths, cfg.kernel, cfg.dropout).to(device)
    xt = X[tr].astype(np.float32)
    model.mu.copy_(torch.tensor(xt.mean((0, 1))).view(1, -1, 1))          # per-lead statistics of the training set
    model.sd.copy_(torch.tensor(xt.std((0, 1)) + 1e-6).view(1, -1, 1))
    del xt
    batches = TensorBatches({"x": X[tr], "y": Y[tr]}, cfg.batch_size, cfg.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = cosine_warmup(opt, cfg.epochs * len(batches), warmup=0.05)

    def loss_fn(m, b):
        x = b["x"].float().transpose(1, 2)
        if cfg.augment:
            x = augment(x)
        return F.binary_cross_entropy_with_logits(m(x), b["y"])

    def evaluate(m):
        return {"val_macro_auroc": macro_auc(Y[va], predict(m, X, va, device))}
    return fit(model, opt, epochs=cfg.epochs, batches=batches, loss_fn=loss_fn, evaluate=evaluate, sched=sched,
               ckpt=Checkpointer(store, key, f"{cfg.work_dir}/ckpt/ecgnet"), metric="val_macro_auroc",
               patience=cfg.patience, ckpt_every_min=cfg.ckpt_every_min, log_every=0)
