import pytest
import torch
from torch import nn

from torch_numopt import SupervisedLearningObjective
from torch_numopt.curvature import (
    NaiveIdentityCalculator,
    ExactHessianCalculator,
    ExactBlockHessianCalculator,
    GaussNewtonApproximation,
    GaussNewtonBlockApproximation,
    HutchinsonDiagonalApproximation,
)
from torch_numopt.utils import param_flatten, param_reshape_like

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def linear_model():
    return nn.Linear(2, 1, bias=True)


@pytest.fixture
def dataset():
    torch.manual_seed(42)
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    return X, y


@pytest.fixture
def objective(linear_model, dataset):
    X, y = dataset
    obj = SupervisedLearningObjective(linear_model, nn.MSELoss(), optimizer=None, weight_decay=0, batch_size=None)
    obj.set_data(X, y)
    return obj


@pytest.fixture
def batched_objective(linear_model, dataset):
    X, y = dataset
    obj = SupervisedLearningObjective(linear_model, nn.MSELoss(), optimizer=None, weight_decay=0, batch_size=2)
    obj.set_data(X, y)
    return obj


@pytest.fixture
def params(objective):
    return tuple(objective.params)


# ----------------------------------------------------------------------
# Analytical helpers
# ----------------------------------------------------------------------


def analytical_hessian_linear(X, y):
    N, d = X.shape
    X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)
    H = (2.0 / N) * X_aug.T @ X_aug
    return H


def flatten_tensors(tensors):
    return torch.cat([t.flatten() for t in tensors])


def block_diag_from_blocks(blocks):
    return torch.block_diag(*blocks)


def extract_diagonal_blocks(H, param_shapes):
    """
    Given a full Hessian matrix H and a list of parameter shapes,
    extract the diagonal blocks corresponding to each parameter group.
    Returns a tuple of blocks.
    """
    blocks = []
    idx = 0
    for shape in param_shapes:
        size = shape.numel() if hasattr(shape, "numel") else torch.prod(torch.tensor(shape)).item()
        block = H[idx : idx + size, idx : idx + size]
        blocks.append(block)
        idx += size
    return tuple(blocks)


# ----------------------------------------------------------------------
# Tests for NaiveIdentityCalculator
# ----------------------------------------------------------------------


def test_naive_identity_scaling(objective, params):
    est = NaiveIdentityCalculator()
    s = est.scaling_matrix(objective, params)
    assert s == 1.0


def test_naive_identity_hvp(objective, params):
    est = NaiveIdentityCalculator()
    v = tuple(torch.randn_like(p) for p in params)  # tuple
    hv = est.hvp(objective, params, v)
    for a, b in zip(v, hv):
        torch.testing.assert_close(a, b)


def test_naive_identity_quadratic(objective, params):
    est = NaiveIdentityCalculator()
    v = tuple(torch.randn_like(p) for p in params)
    q = est.quadratic_form(objective, params, v)
    expected = sum((vv * vv).sum() for vv in v)
    torch.testing.assert_close(q, expected)


# ----------------------------------------------------------------------
# Tests for ExactHessianCalculator (full matrix)
# ----------------------------------------------------------------------
def test_exact_hessian_scaling(objective, params, dataset):
    X, y = dataset
    est = ExactHessianCalculator(damping=None)
    H_est = est.scaling_matrix(objective, params)
    H_ref = analytical_hessian_linear(X, y)
    torch.testing.assert_close(H_est, H_ref, rtol=1e-5, atol=1e-6)


def test_exact_hessian_hvp(objective, params, dataset):
    X, y = dataset
    est = ExactHessianCalculator(damping=None)
    H_ref = analytical_hessian_linear(X, y)
    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    Hv_ref = H_ref @ v_flat
    Hv_est = est.hvp(objective, params, v)
    Hv_est_flat = flatten_tensors(Hv_est)
    torch.testing.assert_close(Hv_est_flat, Hv_ref, rtol=1e-5, atol=1e-6)


def test_exact_hessian_quadratic(objective, params, dataset):
    X, y = dataset
    est = ExactHessianCalculator(damping=None)
    H_ref = analytical_hessian_linear(X, y)
    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    q_ref = v_flat @ H_ref @ v_flat
    q_est = est.quadratic_form(objective, params, v)
    torch.testing.assert_close(q_est, q_ref, rtol=1e-5, atol=1e-6)


def test_exact_hessian_damping_identity(objective, params, dataset):
    X, y = dataset
    mu = 0.1
    est = ExactHessianCalculator(damping="identity", mu=mu)
    H_est = est.scaling_matrix(objective, params)
    H_ref = analytical_hessian_linear(X, y)
    H_ref_damped = H_ref + mu * torch.eye(H_ref.shape[0], device=H_ref.device)
    torch.testing.assert_close(H_est, H_ref_damped, rtol=1e-5, atol=1e-6)


def test_exact_hessian_damping_fletcher(objective, params, dataset):
    X, y = dataset
    mu = 0.1
    est = ExactHessianCalculator(damping="fletcher", mu=mu)
    H_est = est.scaling_matrix(objective, params)
    H_ref = analytical_hessian_linear(X, y)
    H_ref_damped = H_ref + mu * torch.diag(H_ref.diagonal())
    torch.testing.assert_close(H_est, H_ref_damped, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Tests for ExactBlockHessianCalculator (block diagonal)
# ----------------------------------------------------------------------


def test_exact_block_scaling(objective, params, dataset):
    X, y = dataset
    est = ExactBlockHessianCalculator(damping=None)
    blocks = est.scaling_matrix(objective, params)
    assert len(blocks) == len(params)

    # Build full Hessian from analytical and extract diagonal blocks
    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    expected_blocks = extract_diagonal_blocks(H_ref, param_shapes)

    # Compare each block
    for b_est, b_exp in zip(blocks, expected_blocks):
        torch.testing.assert_close(b_est, b_exp, rtol=1e-5, atol=1e-6)


def test_exact_block_hvp(objective, params, dataset):
    X, y = dataset
    est = ExactBlockHessianCalculator(damping=None)

    # Build block-diagonal Hessian from analytical
    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    blocks = extract_diagonal_blocks(H_ref, param_shapes)
    H_block = block_diag_from_blocks(blocks)

    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    Hv_ref = H_block @ v_flat

    Hv_est = est.hvp(objective, params, v)
    Hv_est_flat = flatten_tensors(Hv_est)
    torch.testing.assert_close(Hv_est_flat, Hv_ref, rtol=1e-5, atol=1e-6)


def test_exact_block_quadratic(objective, params, dataset):
    X, y = dataset
    est = ExactBlockHessianCalculator(damping=None)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    blocks = extract_diagonal_blocks(H_ref, param_shapes)
    H_block = block_diag_from_blocks(blocks)

    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    q_ref = v_flat @ H_block @ v_flat
    q_est = est.quadratic_form(objective, params, v)
    torch.testing.assert_close(q_est, q_ref, rtol=1e-5, atol=1e-6)


def test_exact_block_damping_identity(objective, params, dataset):
    X, y = dataset
    mu = 0.1
    est = ExactBlockHessianCalculator(damping="identity", mu=mu)
    blocks = est.scaling_matrix(objective, params)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    expected_blocks = extract_diagonal_blocks(H_ref, param_shapes)
    # Apply damping to each block
    expected_blocks = tuple(b + mu * torch.eye(b.shape[0], device=b.device) for b in expected_blocks)

    for b_est, b_exp in zip(blocks, expected_blocks):
        torch.testing.assert_close(b_est, b_exp, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Tests for GaussNewtonApproximation (full matrix)
# ----------------------------------------------------------------------


def test_gauss_newton_full_scaling(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonApproximation(damping=None)
    H_gn = est.scaling_matrix(objective, params)
    H_ref = analytical_hessian_linear(X, y)
    torch.testing.assert_close(H_gn, H_ref, rtol=1e-5, atol=1e-6)


def test_gauss_newton_full_hvp(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonApproximation(damping=None)
    H_ref = analytical_hessian_linear(X, y)
    v = tuple(torch.randn_like(p) for p in params)  # fixed: tuple
    v_flat = flatten_tensors(v)
    Hv_ref = H_ref @ v_flat
    Hv_est = est.hvp(objective, params, v)
    Hv_est_flat = flatten_tensors(Hv_est)
    torch.testing.assert_close(Hv_est_flat, Hv_ref, rtol=1e-5, atol=1e-6)


def test_gauss_newton_full_quadratic(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonApproximation(damping=None)
    H_ref = analytical_hessian_linear(X, y)
    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    q_ref = v_flat @ H_ref @ v_flat
    q_est = est.quadratic_form(objective, params, v)
    torch.testing.assert_close(q_est, q_ref, rtol=1e-5, atol=1e-6)


def test_gauss_newton_full_damping_identity(objective, params, dataset):
    X, y = dataset
    mu = 0.1
    est = GaussNewtonApproximation(damping="identity", mu=mu)
    H_gn = est.scaling_matrix(objective, params)
    H_ref = analytical_hessian_linear(X, y)
    H_ref_damped = H_ref + mu * torch.eye(H_ref.shape[0], device=H_ref.device)
    torch.testing.assert_close(H_gn, H_ref_damped, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Tests for GaussNewtonBlockApproximation (block diagonal)
# ----------------------------------------------------------------------


def test_gauss_newton_block_scaling(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonBlockApproximation(damping=None)
    blocks = est.scaling_matrix(objective, params)
    assert len(blocks) == len(params)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    expected_blocks = extract_diagonal_blocks(H_ref, param_shapes)

    for b_est, b_exp in zip(blocks, expected_blocks):
        torch.testing.assert_close(b_est, b_exp, rtol=1e-5, atol=1e-6)


def test_gauss_newton_block_hvp(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonBlockApproximation(damping=None)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    blocks = extract_diagonal_blocks(H_ref, param_shapes)
    H_block = block_diag_from_blocks(blocks)

    v = tuple(torch.randn_like(p) for p in params)  # fixed: tuple
    v_flat = flatten_tensors(v)
    Hv_ref = H_block @ v_flat

    Hv_est = est.hvp(objective, params, v)
    Hv_est_flat = flatten_tensors(Hv_est)
    torch.testing.assert_close(Hv_est_flat, Hv_ref, rtol=1e-5, atol=1e-6)


def test_gauss_newton_block_quadratic(objective, params, dataset):
    X, y = dataset
    est = GaussNewtonBlockApproximation(damping=None)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    blocks = extract_diagonal_blocks(H_ref, param_shapes)
    H_block = block_diag_from_blocks(blocks)

    v = tuple(torch.randn_like(p) for p in params)
    v_flat = flatten_tensors(v)
    q_ref = v_flat @ H_block @ v_flat
    q_est = est.quadratic_form(objective, params, v)
    torch.testing.assert_close(q_est, q_ref, rtol=1e-5, atol=1e-6)


def test_gauss_newton_block_damping_identity(objective, params, dataset):
    X, y = dataset
    mu = 0.1
    est = GaussNewtonBlockApproximation(damping="identity", mu=mu)
    blocks = est.scaling_matrix(objective, params)

    H_ref = analytical_hessian_linear(X, y)
    param_shapes = [p.shape for p in params]
    expected_blocks = extract_diagonal_blocks(H_ref, param_shapes)
    expected_blocks = tuple(b + mu * torch.eye(b.shape[0], device=b.device) for b in expected_blocks)

    for b_est, b_exp in zip(blocks, expected_blocks):
        torch.testing.assert_close(b_est, b_exp, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Tests for HutchinsonDiagonalApproximation
# ----------------------------------------------------------------------


def test_hutchinson_diagonal_shape(objective, params):
    est = HutchinsonDiagonalApproximation(n_samples=5)
    diag_blocks = est.scaling_matrix(objective, params)
    assert len(diag_blocks) == len(params)
    for d, p in zip(diag_blocks, params):
        assert d.shape == p.shape


def test_hutchinson_diagonal_against_true_diagonal(objective, params, dataset):
    X, y = dataset
    H_ref = analytical_hessian_linear(X, y)
    true_diag = H_ref.diagonal()
    est = HutchinsonDiagonalApproximation(n_samples=50)
    diag_blocks = est.scaling_matrix(objective, params)
    diag_flat = flatten_tensors(diag_blocks)
    torch.testing.assert_close(diag_flat, true_diag, rtol=0.2, atol=0.1)


# ----------------------------------------------------------------------
# Tests for batched vs non‑batched consistency (these are expected to fail)
# ----------------------------------------------------------------------


def test_exact_hessian_batched_consistency(objective, batched_objective, params):
    est = ExactHessianCalculator(damping=None)
    H_full = est.scaling_matrix(objective, params)
    H_batch = est.scaling_matrix(batched_objective, params)
    torch.testing.assert_close(H_full, H_batch, rtol=1e-5, atol=1e-6)


def test_gauss_newton_batched_consistency(objective, batched_objective, params):
    est = GaussNewtonApproximation(damping=None)
    H_full = est.scaling_matrix(objective, params)
    H_batch = est.scaling_matrix(batched_objective, params)
    torch.testing.assert_close(H_full, H_batch, rtol=1e-5, atol=1e-6)


def test_exact_block_batched_consistency(objective, batched_objective, params):
    est = ExactBlockHessianCalculator(damping=None)
    blocks_full = est.scaling_matrix(objective, params)
    blocks_batch = est.scaling_matrix(batched_objective, params)
    H_full = block_diag_from_blocks(blocks_full)
    H_batch = block_diag_from_blocks(blocks_batch)
    torch.testing.assert_close(H_full, H_batch, rtol=1e-5, atol=1e-6)


def test_gauss_newton_block_batched_consistency(objective, batched_objective, params):
    est = GaussNewtonBlockApproximation(damping=None)
    blocks_full = est.scaling_matrix(objective, params)
    blocks_batch = est.scaling_matrix(batched_objective, params)
    H_full = block_diag_from_blocks(blocks_full)
    H_batch = block_diag_from_blocks(blocks_batch)
    torch.testing.assert_close(H_full, H_batch, rtol=1e-5, atol=1e-6)
