"""Resumable PyTorch training.

Everything needed to continue is saved as `last.pt` (model, optimizer, scheduler, RNG states, epoch AND the step
inside the epoch) every `ckpt_every_min` minutes and at the end of every epoch, locally and in the bucket.
The batch order of an epoch depends only on (seed, epoch), so a new Kaggle session skips exactly the batches that
were already done and continues in the middle of the epoch. `best.pt` keeps the best validation weights.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, store, key: str, local_dir):
        self.store, self.key, self.dir = store, key.strip("/"), Path(local_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, obj, name: str):
        f = self.dir / name
        tmp = f.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(f)
        if self.store is not None:
            self.store.put(f, f"{self.key}/{name}")

    def load(self, name: str):
        f = self.dir / name
        if not f.exists() and self.store is not None:
            self.store.get(f"{self.key}/{name}", f)
        return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None


@dataclass
class State:
    epoch: int = 0                 # finished epochs
    step: int = 0                  # finished steps inside the current epoch
    loss_sum: float = 0.0
    loss_n: int = 0
    best: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0
    done: bool = False
    history: list = field(default_factory=list)


def _rng_state():
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class TensorBatches:
    """Mini-batches from in-memory arrays. Order = f(seed, epoch) -> exact mid-epoch resume.
    With `on_device=True` the arrays are copied to the GPU once (fast for tabular data)."""

    def __init__(self, arrays: dict, batch_size: int, seed: int, device, on_device: bool = True, index=None,
                 tensors: dict | None = None):
        """`index`: use only these rows (no copy of the big arrays). `tensors`: already-built torch tensors."""
        self.bs, self.seed, self.device = batch_size, seed, torch.device(device)
        tgt = self.device if on_device else torch.device("cpu")
        def to_t(v):                                   # numpy array OR torch tensor (CPU or GPU)
            if isinstance(v, torch.Tensor):
                return v.to(tgt)
            return torch.as_tensor(np.ascontiguousarray(v)).to(tgt)
        self.t = tensors if tensors is not None else {k: to_t(v) for k, v in arrays.items()}
        some = next(iter(self.t.values()))
        self.index = None if index is None else torch.as_tensor(np.asarray(index, dtype=np.int64)).to(some.device)
        self.n = len(some) if index is None else len(self.index)
        self._perm = (None, None)

    def __len__(self):
        return max(1, -(-self.n // self.bs))

    def __call__(self, epoch: int, step: int) -> dict:
        if self._perm[0] != epoch:
            self._perm = (epoch, torch.from_numpy(np.random.default_rng(self.seed + epoch).permutation(self.n)))
        idx = self._perm[1][step * self.bs:(step + 1) * self.bs]
        some = next(iter(self.t.values()))
        idx = idx.to(some.device)
        if self.index is not None:
            idx = self.index[idx]
        return {k: v[idx].to(self.device, non_blocking=True) for k, v in self.t.items()}


def fit(model, opt, *, epochs: int, batches, loss_fn, evaluate, ckpt: Checkpointer, metric: str,
        sched=None, patience: int = 0, ckpt_every_min: float = 10.0, grad_clip: float = 1.0, log_every: int = 500):
    """Train with resume. `evaluate(model) -> dict` must contain `metric` (higher is better).
    Returns (model with best weights, State)."""
    session = uuid.uuid4().hex[:6]
    st = State()
    last = ckpt.load("last.pt")
    if last is not None:
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["opt"])
        if sched is not None and last.get("sched") is not None:
            sched.load_state_dict(last["sched"])
        st = State(**last["state"])
        _set_rng(last["rng"])
        log.info("RESUME from bucket checkpoint: %d epoch(s) done, %d/%d steps of epoch %d already done",
                 st.epoch, st.step, len(batches), st.epoch + 1)

    def save_last():
        ckpt.save({"model": model.state_dict(), "opt": opt.state_dict(),
                   "sched": sched.state_dict() if sched is not None else None,
                   "state": asdict(st), "rng": _rng_state()}, "last.pt")

    t_save = time.time()
    while not st.done and st.epoch < epochs:
        model.train()
        t_ep = time.time()
        for step in range(st.step, len(batches)):
            b = batches(st.epoch, step)
            loss = loss_fn(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            st.step, st.loss_sum, st.loss_n = step + 1, st.loss_sum + float(loss.detach()), st.loss_n + 1
            if log_every and (step + 1) % log_every == 0:
                log.info("epoch %d step %d/%d loss %.4f", st.epoch + 1, step + 1, len(batches), st.loss_sum / st.loss_n)
            if time.time() - t_save > ckpt_every_min * 60:
                save_last()
                t_save = time.time()
                log.info("mid-epoch checkpoint saved (epoch %d, step %d)", st.epoch + 1, st.step)
        model.eval()
        with torch.no_grad():
            val = {k: float(v) for k, v in evaluate(model).items()}
        row = {"epoch": st.epoch + 1, "train_loss": st.loss_sum / max(st.loss_n, 1), **val,
               "minutes": round((time.time() - t_ep) / 60, 2), "session": session}
        st.history.append(row)
        if val[metric] > st.best:
            st.best, st.best_epoch, st.bad_epochs = val[metric], st.epoch + 1, 0
            ckpt.save({"model": model.state_dict(), "epoch": st.epoch + 1, metric: val[metric]}, "best.pt")
        else:
            st.bad_epochs += 1
        log.info("epoch %d: %s", st.epoch + 1, ", ".join(f"{k}={v:.4f}" for k, v in row.items()
                                                       if isinstance(v, float)))
        st.epoch, st.step, st.loss_sum, st.loss_n = st.epoch + 1, 0, 0.0, 0
        if patience and st.bad_epochs >= patience:
            log.info("early stop: no %s improvement for %d epochs", metric, patience)
            st.done = True
        save_last()
        t_save = time.time()
    st.done = True
    save_last()
    best = ckpt.load("best.pt")
    if best is not None:
        model.load_state_dict(best["model"])
    model.eval()
    st.session = session
    return model, st


def history(st: State, metric: str) -> pd.DataFrame:
    """One row per epoch; `source` tells whether the epoch was trained now or restored from the bucket."""
    df = pd.DataFrame(st.history)
    if not len(df):
        return pd.DataFrame({"info": ["no epochs"]})
    cur = getattr(st, "session", None)
    df["source"] = np.where(df.session == cur, "this session", "checkpoint")
    df["best"] = np.where(df.epoch == st.best_epoch, "★", "")
    return df.drop(columns="session")


def cosine_warmup(opt, total_steps: int, warmup: float = 0.03):
    w = max(1, int(total_steps * warmup))

    def f(s):
        if s < w:
            return (s + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - w) / max(1, total_steps - w))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)
