"""Tests for linear probes."""
import torch

from mechreward.probes.linear_probe import LinearProbe, TorchLinearProbe
from mechreward.probes.training import train_linear_probe


def test_linear_probe_predict_2d():
    weight = torch.tensor([1.0, -1.0, 0.5])
    bias = torch.tensor(0.0)
    probe = LinearProbe(weight=weight, bias=bias, d_model=3)

    hidden = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = probe.predict(hidden)
    assert torch.allclose(out, torch.tensor([1.0, -1.0, 0.5]))


def test_linear_probe_predict_3d():
    weight = torch.tensor([1.0, 0.0])
    bias = torch.tensor(0.0)
    probe = LinearProbe(weight=weight, bias=bias, d_model=2)

    hidden = torch.tensor([[[1.0, 5.0], [2.0, 5.0]]])
    out = probe.predict(hidden)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0]]))


def test_torch_probe_to_static_roundtrip():
    tp = TorchLinearProbe(d_model=8)
    static = tp.to_static(name="mine")
    assert static.d_model == 8
    assert static.name == "mine"
    assert static.weight.shape == (8,)


def test_train_linear_probe_learns_easy_task():
    torch.manual_seed(0)
    n = 200
    d = 32
    # Class 0: mean 0; Class 1: mean 3
    x_0 = torch.randn(n // 2, d)
    x_1 = torch.randn(n // 2, d) + 3.0
    x = torch.cat([x_0, x_1], dim=0)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)], dim=0)

    result = train_linear_probe(x, y, epochs=100, lr=1e-2, verbose=False, device="cpu")
    assert result.train_accuracy > 0.95
