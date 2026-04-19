import pytest
import torch
from torch_numopt.utils import (
    param_sizes,
    param_reshape_like,
    param_flatten,
    fix_stability,
    pinv_svd_trunc,
)

def test_param_sizes():
    params = [torch.randn(3, 4), torch.randn(5)]
    sizes = param_sizes(params)
    assert sizes == [(3, 4), (5,)]

def test_param_reshape_like():
    params = [torch.randn(2, 3), torch.randn(4)]
    flat = torch.arange(10).float()
    reshaped = param_reshape_like(flat, params)
    assert len(reshaped) == 2
    assert reshaped[0].shape == (2, 3)
    assert reshaped[1].shape == (4,)
    assert torch.allclose(reshaped[0].flatten(), flat[:6])
    assert torch.allclose(reshaped[1].flatten(), flat[6:10])

def test_param_flatten():
    params = [torch.tensor([1.0, 2.0]), torch.tensor([[3.0, 4.0]])]
    flat = param_flatten(params)
    assert torch.allclose(flat, torch.tensor([1., 2., 3., 4.]))

def test_fix_stability():
    A = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]])
    A_fixed = fix_stability(A)
    assert torch.allclose(A_fixed, A + torch.eye(2) * torch.finfo(A.dtype).eps)

def test_pinv_svd_trunc():
    A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    A_pinv = pinv_svd_trunc(A, thresh=1e-4)
    # Should be close to analytical pseudoinverse
    expected = torch.tensor([[0.04, 0.08], [0.08, 0.16]])
    assert torch.allclose(A_pinv, expected, atol=1e-5)