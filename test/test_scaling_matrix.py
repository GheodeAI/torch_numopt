import torch
import pytest
from torch_numopt.scaling_matrix_calculator import (
    NaiveIdentityCalculator,
    ExactBlockHessianCalculator,
    GaussNewtonBlockApproximation,
    HutchinsonDiagonalApproximation,
)

@pytest.mark.parametrize("calc_class, kwargs", [
    (NaiveIdentityCalculator, {}),
    (ExactBlockHessianCalculator, {"damping": None}),
    (ExactBlockHessianCalculator, {"damping": "identity", "mu": 0.1}),
    (GaussNewtonBlockApproximation, {"damping": None}),
    (GaussNewtonBlockApproximation, {"damping": "identity", "mu": 0.1}),
    (HutchinsonDiagonalApproximation, {"n_samples": 3}),
])
def test_scaling_matrix_calculator(model_and_data, calc_class, kwargs):
    model, X, y, loss_fn = model_and_data
    calc = calc_class(model, **kwargs)
    
    # For Hutchinson, we need gradients first
    if isinstance(calc, HutchinsonDiagonalApproximation):
        # Perform a forward/backward to have grad available? Not needed, calc uses functional call.
        pass

    result = calc(X, y, loss_fn)
    
    if isinstance(calc, NaiveIdentityCalculator):
        assert result is None
    elif isinstance(calc, HutchinsonDiagonalApproximation):
        # Should return list of diagonal tensors with same shapes as parameters
        params = list(model.parameters())
        assert len(result) == len(params)
        for r, p in zip(result, params):
            assert r.shape == p.shape
    else:
        # Should be list of square matrices (Hessian blocks)
        params = list(model.parameters())
        assert len(result) == len(params)
        for r, p in zip(result, params):
            n_params = p.numel()
            assert r.shape == (n_params, n_params)
            # Symmetry check (approximate)
            assert torch.allclose(r, r.T, atol=1e-5)