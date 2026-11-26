"""Mid-epoch crash -> resume gives exactly the same model as an uninterrupted run."""
import numpy as np
import pytest
import torch

from src.store import LocalStore
from src.trainer import Checkpointer, TensorBatches, fit, history


class Crash(Exception):
    pass


def _setup(seed=0):
    torch.manual_seed(seed)
    r = np.random.default_rng(0)
    X = r.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] + r.normal(size=600) > 0).astype(np.float32)
    m = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    return X, y, m, opt


def _train(tmp, crash_at=None, name="a"):
    X, y, m, opt = _setup()
    b = TensorBatches({"x": X, "y": y}, 64, 0, "cpu")
    calls = {"n": 0}

    def batches(ep, st):
        calls["n"] += 1
        if crash_at and (ep, st) == crash_at and calls["n"] < 100:
            raise Crash
        return b(ep, st)
    batches.__len__ = None
    wrapped = type("B", (), {"__len__": lambda s: len(b), "__call__": lambda s, e, t: batches(e, t)})()
    loss = lambda mm, bb: torch.nn.functional.binary_cross_entropy_with_logits(mm(bb["x"]).squeeze(-1), bb["y"])
    ev = lambda mm: {"acc": float(((mm(torch.tensor(X)).squeeze(-1) > 0).float() == torch.tensor(y)).float().mean())}
    ck = Checkpointer(LocalStore(tmp / f"bucket_{name}"), "ck", tmp / f"local_{name}")
    return fit(m, opt, epochs=3, batches=wrapped, loss_fn=loss, evaluate=ev, ckpt=ck, metric="acc",
               ckpt_every_min=0, log_every=0), ck


def test_mid_epoch_resume_is_exact(tmp_path):
    (m_ref, st_ref), _ = _train(tmp_path, name="ref")
    with pytest.raises(Crash):
        _train(tmp_path, crash_at=(1, 4), name="crash")
    # new "session": empty local folder, same bucket
    import shutil
    shutil.rmtree(tmp_path / "local_crash")
    (m2, st2), _ = _train(tmp_path, name="crash")
    for a, b in zip(m_ref.state_dict().values(), m2.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)
    h = history(st2, "acc")
    assert list(h.epoch) == [1, 2, 3] and h.source.tolist() == ["checkpoint", "this session", "this session"]
    assert (h.best == "★").sum() == 1


def test_tensor_batches_accepts_torch_tensors():
    """Regression: data already on the device (torch tensors, e.g. cuda) must not go through numpy."""
    X = torch.randn(100, 3)
    b = TensorBatches({"x": X[torch.arange(50)], "y": torch.ones(50)}, 16, 0, "cpu")
    assert len(b) == 4 and b(0, 0)["x"].shape == (16, 3)
    b2 = TensorBatches({"x": X}, 16, 0, "cpu", index=np.arange(10))
    assert set(b2(0, 0)["x"].sum(1).tolist()) <= set(X[:10].sum(1).tolist())
