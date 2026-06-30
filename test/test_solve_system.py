import pytest
import torch
from torch_numopt.solve_system import solve_system
from torch_numopt.curvature_estimator import CurvatureEstimator
from torch_numopt.objective import ObjectiveFunction
from torch_numopt.utils import param_flatten, param_reshape_like, param_zero_like, Params


# ---------- Mock Curvature Estimators ----------
class ScalarCurvature(CurvatureEstimator):
    """Curvature estimator that returns a scalar (ndim=0)."""

    def __init__(self, scalar=2.0):
        super().__init__(ndim=0, uses_blocks=False)
        self.scalar = scalar

    def scaling_matrix(self, objective, params):
        return torch.tensor(self.scalar, device=params[0].device, dtype=params[0].dtype)

    def hvp(self, objective, params, step_dir):
        return tuple(self.scalar * p for p in step_dir)

    def quadratic_form(self, objective, params, step_dir):
        return self.scalar * sum(torch.sum(p * p) for p in step_dir)


class DiagonalCurvature(CurvatureEstimator):
    """Curvature estimator that returns a diagonal matrix (ndim=1)."""

    def __init__(self, diag_values):
        super().__init__(ndim=1, uses_blocks=False)
        self.diag_values = diag_values  # tuple of tensors (one per param group)

    def scaling_matrix(self, objective, params):
        return tuple(d.clone().detach() for d in self.diag_values)

    def hvp(self, objective, params, step_dir):
        return tuple(d * p for d, p in zip(self.diag_values, step_dir))

    def quadratic_form(self, objective, params, step_dir):
        return sum(torch.sum(d * p * p) for d, p in zip(self.diag_values, step_dir))


class FullMatrixCurvature(CurvatureEstimator):
    """Curvature estimator that returns a full matrix (ndim=2)."""

    def __init__(self, matrix):
        super().__init__(ndim=2, uses_blocks=False)
        self.matrix = matrix  # 2D tensor

    def scaling_matrix(self, objective, params):
        return self.matrix

    def hvp(self, objective, params, step_dir):
        flat = torch.cat([p.flatten() for p in step_dir])
        result_flat = self.matrix @ flat
        return param_reshape_like(result_flat, step_dir)

    def quadratic_form(self, objective, params, step_dir):
        flat = torch.cat([p.flatten() for p in step_dir])
        return flat @ self.matrix @ flat


class BlockDiagonalCurvature(CurvatureEstimator):
    """Curvature estimator that returns a block-diagonal matrix (ndim=2, uses_blocks=True)."""

    def __init__(self, blocks):
        super().__init__(ndim=2, uses_blocks=True)
        self.blocks = blocks  # tuple of 2D tensors (one per parameter group)

    def scaling_matrix(self, objective, params):
        return self.blocks

    def hvp(self, objective, params, step_dir):
        result = []
        for B, p in zip(self.blocks, step_dir):
            result.append((B @ p.flatten()).reshape(p.shape))
        return tuple(result)

    def quadratic_form(self, objective, params, step_dir):
        total = 0.0
        for B, p in zip(self.blocks, step_dir):
            flat_p = p.flatten()
            total += flat_p @ B @ flat_p
        return total


# ---------- Fixtures ----------
@pytest.fixture
def dummy_objective():
    """A minimal ObjectiveFunction that provides params and loss (unused)."""

    class DummyObjective(ObjectiveFunction):
        def __init__(self, params):
            super().__init__(params, optimizer=None, batched=False)

        def loss(self, *params, batch_idx=None):
            return torch.tensor(0.0)

        def residual(self, *params, batch_idx=None):
            raise NotImplementedError

    params = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))  # two groups
    return DummyObjective(params)


# ---------- Tests for Scalar Curvature ----------
def test_solve_system_scalar(dummy_objective):
    """Scalar curvature: B = scalar * I, solution should be rhs / scalar."""
    scalar = 2.0
    curv = ScalarCurvature(scalar)
    rhs = (torch.tensor([4.0, 6.0]), torch.tensor([8.0, 10.0]))
    x = solve_system(curv, dummy_objective, rhs, solver="solve")
    expected = tuple(r / scalar for r in rhs)
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei)


# ---------- Tests for Diagonal Curvature ----------
def test_solve_system_diagonal(dummy_objective):
    """Diagonal curvature: B is diagonal, solution should be rhs / diag."""
    diag = (torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
    curv = DiagonalCurvature(diag)
    rhs = (torch.tensor([4.0, 9.0]), torch.tensor([16.0, 25.0]))
    x = solve_system(curv, dummy_objective, rhs, solver="solve")
    expected = tuple(r / d for r, d in zip(rhs, diag))
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei)


# ---------- Tests for Full Matrix Curvature ----------
def test_solve_system_full_matrix(dummy_objective):
    """Full matrix: B is a 4x4 tensor, solve B*x = rhs."""
    # dummy_objective has two groups of size 2 each -> total 4 params
    B = torch.tensor([[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0], [0.0, 0.0, 5.0, 2.0], [0.0, 0.0, 2.0, 4.0]], dtype=torch.float32)
    curv = FullMatrixCurvature(B)

    x_flat = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rhs_flat = B @ x_flat
    rhs = (rhs_flat[0:2], rhs_flat[2:4])

    x = solve_system(curv, dummy_objective, rhs, solver="solve")

    expected = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei, rtol=1e-5)


# ---------- Tests for Block‑Diagonal Curvature ----------
def test_solve_system_block_diagonal(dummy_objective):
    """Block diagonal: each block is a matrix."""
    B1 = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    B2 = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    curv = BlockDiagonalCurvature((B1, B2))
    rhs = (torch.tensor([2.0, 1.0]), torch.tensor([0.0, 4.0]))
    x = solve_system(curv, dummy_objective, rhs, solver="solve")
    expected = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei)


# ---------- Direct Solvers (all variants) ----------
@pytest.mark.parametrize("solver", ["solve", "pinv", "pinv-trunc", "lsqrs", "safe-lsqrs", "cholesky"])
def test_solve_system_all_direct_solvers(solver, dummy_objective):
    """Test all direct solvers on a simple block-diagonal problem."""
    B1 = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    B2 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    curv = BlockDiagonalCurvature((B1, B2))
    rhs = (torch.tensor([5.0, 4.0]), torch.tensor([4.0, 6.0]))
    x = solve_system(curv, dummy_objective, rhs, solver=solver)
    expected = (torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0]))
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei, rtol=1e-5)


# ---------- Iterative Solvers (CG variants) ----------
@pytest.mark.parametrize("solver", ["cg", "cg-trunc", "cr"])
def test_solve_system_iterative_solvers(solver, dummy_objective):
    """Test iterative solvers on SPD block-diagonal matrices."""
    B1 = torch.tensor([[5.0, 1.0], [1.0, 4.0]])  # SPD
    B2 = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    curv = BlockDiagonalCurvature((B1, B2))
    rhs = (torch.tensor([6.0, 5.0]), torch.tensor([6.0, 9.0]))
    x = solve_system(curv, dummy_objective, rhs, solver=solver, max_iter=100, tol=1e-8)
    expected = (torch.tensor([1.0, 1.0]), torch.tensor([2.0, 3.0]))
    for xi, ei in zip(x, expected):
        assert torch.allclose(xi, ei, rtol=1e-4, atol=1e-5)


# ---------- Edge Cases and Error Handling ----------
def test_solve_system_unsupported_solver(dummy_objective):
    """Unsupported solver should raise AssertionError."""
    curv = ScalarCurvature()
    rhs = (torch.tensor([1.0]),)
    with pytest.raises(AssertionError):
        solve_system(curv, dummy_objective, rhs, solver="invalid_solver")


def test_solve_system_fallback_solve_to_lsqrs(dummy_objective):
    """When solver='solve' fails (singular matrix), fallback to lsqrs gives least‑squares solution."""
    B1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # singular
    B2 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    curv = BlockDiagonalCurvature((B1, B2))
    rhs = (torch.tensor([2.0, 2.0]), torch.tensor([4.0, 6.0]))

    x = solve_system(curv, dummy_objective, rhs, solver="solve")

    Bx1 = B1 @ x[0].flatten()
    Bx2 = B2 @ x[1].flatten()
    residual1 = torch.norm(Bx1 - rhs[0])
    residual2 = torch.norm(Bx2 - rhs[1])
    # The singular block's RHS is in the column space, so residual should be near zero.
    assert residual1 < 1e-6
    assert residual2 < 1e-6
    # Also verify that the solution is not simply the RHS (which would be a trivial fallback).
    assert not torch.allclose(x[0], rhs[0])
