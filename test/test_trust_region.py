import pytest
import torch
from torch_numopt.trust_region import (
    CauchyPointTRSolver,
    DoglegTRSolver,
    ExactTRSolver,
    create_trust_region_solver,
)
from torch_numopt.curvature import (
    ExactHessianCalculator,
    GaussNewtonBlockApproximation,
    NaiveIdentityCalculator,
)
from torch_numopt.objective import ObjectiveFunction
from torch_numopt.utils import param_dot, param_norm, param_scaled_add, param_diff

# ----------------------------------------------------------------------
# Helper: Quadratic objective (same as for line search)
# ----------------------------------------------------------------------


class QuadraticObjective(ObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        super().__init__(params=(torch.zeros_like(b),), optimizer=None, batched=False)

    def loss(self, *params, batch_idx=None):
        x = params[0]
        return 0.5 * x @ self.A @ x - self.b @ x

    def residual(self, *params, batch_idx=None):
        raise NotImplementedError


def scalar_quadratic(a, b):
    A = torch.tensor([[a]], dtype=torch.float64)
    b_t = torch.tensor([b], dtype=torch.float64)
    return QuadraticObjective(A, b_t)


def diag_quadratic(diag_a, b):
    A = torch.diag(torch.tensor(diag_a, dtype=torch.float64))
    b_t = torch.tensor(b, dtype=torch.float64)
    return QuadraticObjective(A, b_t)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def scalar_obj():
    # f(x) = 0.5 * 2 * x^2 - 3*x => gradient at 0 is -3
    return scalar_quadratic(a=2.0, b=3.0)


@pytest.fixture
def scalar_params():
    return (torch.tensor([0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def scalar_grad(scalar_obj, scalar_params):
    # gradient = A*x - b = -3
    return (torch.tensor([-3.0], dtype=torch.float64),)


@pytest.fixture
def diag_obj():
    # A = diag(2,1), b = [1,2] => gradient at 0 is [-1, -2]
    return diag_quadratic(diag_a=[2.0, 1.0], b=[1.0, 2.0])


@pytest.fixture
def diag_params():
    return (torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def diag_grad(diag_obj, diag_params):
    return (torch.tensor([-1.0, -2.0], dtype=torch.float64),)


# For exact Hessian, we need a curvature estimator that returns the full Hessian.
@pytest.fixture
def exact_curvature():
    return ExactHessianCalculator(damping=None)


@pytest.fixture
def naive_curvature():
    return NaiveIdentityCalculator()


# ----------------------------------------------------------------------
# Analytical solutions for trust-region subproblem (2D)
# ----------------------------------------------------------------------


def analytical_cauchy_point(g, H, radius):
    """
    Cauchy point for quadratic: p_c = - (g^T g) / (g^T H g) * g  if g^T H g > 0,
    otherwise p = - radius * g / ||g||.
    """
    g_norm = torch.norm(g)
    if g_norm == 0:
        return torch.zeros_like(g)
    gHg = g @ H @ g
    if gHg > 0:
        alpha = (g @ g) / gHg
        p = -alpha * g
        if torch.norm(p) <= radius:
            return p
    # If Cauchy point outside radius, scale to boundary
    return -radius * g / g_norm


def analytical_newton_step(g, H):
    """Newton step: -H^{-1} g, assumes H positive definite."""
    return -torch.linalg.solve(H, g)


def analytical_dogleg(g, H, radius):
    """Dogleg for positive definite H."""
    p_gn = analytical_newton_step(g, H)
    if torch.norm(p_gn) <= radius:
        return p_gn
    # Cauchy point (scaled to boundary if needed)
    p_c = analytical_cauchy_point(g, H, radius)
    # If Cauchy point is already at boundary, return it
    if torch.norm(p_c) >= radius - 1e-12:
        return p_c
    # Otherwise, interpolate between Cauchy and Newton
    # Find the intersection of the dogleg path with the circle
    # Path: p(t) = t * p_c + (1 - t) * p_gn, t in [0,1]
    # Solve ||p(t)||^2 = radius^2
    d = p_c - p_gn
    a = torch.dot(d, d)
    b = 2 * torch.dot(p_gn, d)
    c = torch.dot(p_gn, p_gn) - radius**2
    t = (-b + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)  # the positive root
    t = torch.clamp(t, 0, 1)
    return t * p_c + (1 - t) * p_gn


# ----------------------------------------------------------------------
# Tests for CauchyPointTRSolver
# ----------------------------------------------------------------------


def test_cauchy_point_scalar(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    radius = 0.5
    solver = CauchyPointTRSolver(curvature_estimator=exact_curvature)
    step_dir = solver.optimize_model(scalar_obj, scalar_params, radius, scalar_grad)

    # Step must be within radius
    assert param_norm(step_dir) <= radius + 1e-10

    # For scalar, gradient = -3, H = 2. Cauchy point: alpha = (9)/(9*2? wait gHg = (-3)^2 * 2 = 18, g^T g = 9 => alpha=0.5, p = -0.5*(-3)=1.5? Wait sign: p = -alpha * g = -0.5 * (-3) = 1.5. But with radius 0.5, the unconstrained Cauchy point is outside, so we scale to boundary: p = -radius * g / ||g|| = -0.5 * (-3)/3 = 0.5.
    # So step_dir should be 0.5.
    expected = torch.tensor([0.5], dtype=torch.float64)
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-5, atol=1e-6)

    # Check descent: g·p < 0
    assert param_dot(scalar_grad, step_dir) < 0


def test_cauchy_point_2d(diag_obj, diag_params, diag_grad, exact_curvature):
    radius = 0.8
    solver = CauchyPointTRSolver(curvature_estimator=exact_curvature)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    assert param_norm(step_dir) <= radius + 1e-10
    # Analytical Cauchy: g = [-1, -2], H = diag(2,1)
    g = diag_grad[0]
    H = exact_curvature.scaling_matrix(diag_obj, diag_params)  # full Hessian
    p_c = analytical_cauchy_point(g, H, radius)
    torch.testing.assert_close(step_dir[0], p_c, rtol=1e-5, atol=1e-6)
    assert param_dot(diag_grad, step_dir) < 0


# ----------------------------------------------------------------------
# Tests for DoglegTRSolver
# ----------------------------------------------------------------------


def test_dogleg_scalar_large_radius(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    # Radius large enough to contain Newton step
    radius = 10.0
    solver = DoglegTRSolver(curvature_estimator=exact_curvature)
    step_dir = solver.optimize_model(scalar_obj, scalar_params, radius, scalar_grad)

    # Newton step: -H^{-1} g = - (1/2)*(-3) = 1.5
    expected = torch.tensor([1.5], dtype=torch.float64)
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-5, atol=1e-6)
    assert param_norm(step_dir) <= radius


def test_dogleg_scalar_small_radius(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    radius = 0.3
    solver = DoglegTRSolver(curvature_estimator=exact_curvature)
    step_dir = solver.optimize_model(scalar_obj, scalar_params, radius, scalar_grad)

    # For small radius, dogleg = Cauchy point scaled to boundary = -radius * g / ||g|| = -0.3*(-3)/3 = 0.3
    expected = torch.tensor([0.3], dtype=torch.float64)
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-5, atol=1e-6)
    assert param_norm(step_dir) <= radius + 1e-10


def test_dogleg_2d(diag_obj, diag_params, diag_grad, exact_curvature):
    radius = 1.2
    solver = DoglegTRSolver(curvature_estimator=exact_curvature)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    g = diag_grad[0]
    H = exact_curvature.scaling_matrix(diag_obj, diag_params)
    p_analytical = analytical_dogleg(g, H, radius)
    torch.testing.assert_close(step_dir[0], p_analytical, rtol=1e-5, atol=1e-6)
    assert param_norm(step_dir) <= radius + 1e-10
    assert param_dot(diag_grad, step_dir) < 0


def test_dogleg_non_positive_definite(diag_obj, diag_params, diag_grad):
    # Use a Hessian with negative eigenvalue: we can't use exact_curvature because it's positive definite.
    # We'll construct a curvature estimator that returns a matrix with a negative eigenvalue.
    class NonPDCurvature(ExactHessianCalculator):
        def scaling_matrix(self, objective, params):
            H = super().scaling_matrix(objective, params)
            # Make it non-PD by subtracting 3 from first diagonal element
            H[0, 0] -= 3.0  # now first eigenvalue = -1 (since original was 2)
            return H

    curv = NonPDCurvature(damping=None)
    radius = 1.0
    solver = DoglegTRSolver(curvature_estimator=curv)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    # Should still produce a descent direction and stay within radius
    assert param_norm(step_dir) <= radius + 1e-10
    assert param_dot(diag_grad, step_dir) < 0


# ----------------------------------------------------------------------
# Tests for ExactTRSolver
# ----------------------------------------------------------------------


def test_exact_tr_scalar(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    radius = 1.0
    solver = ExactTRSolver(curvature_estimator=exact_curvature, iters=20, tol=1e-10)
    step_dir = solver.optimize_model(scalar_obj, scalar_params, radius, scalar_grad)

    # Exact solution for scalar: if Newton step (1.5) > radius, solution is on boundary: p = radius * sign(-g) = 1.0
    expected = torch.tensor([1.0], dtype=torch.float64)
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-5, atol=1e-6)
    assert param_norm(step_dir) <= radius + 1e-10
    assert param_dot(scalar_grad, step_dir) < 0


def test_exact_tr_2d_large_radius(diag_obj, diag_params, diag_grad, exact_curvature):
    radius = 10.0
    solver = ExactTRSolver(curvature_estimator=exact_curvature, iters=20, tol=1e-10)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    # Newton step: -H^{-1}g = [0.5, 2]
    expected = torch.tensor([0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-5, atol=1e-6)
    assert param_norm(step_dir) <= radius + 1e-10


def test_exact_tr_2d_small_radius(diag_obj, diag_params, diag_grad, exact_curvature):
    radius = 1e-4
    solver = ExactTRSolver(curvature_estimator=exact_curvature, iters=20, tol=1e-10)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    # For small radius, exact solution = Cauchy point on boundary = -radius * g / ||g||
    g = diag_grad[0]
    g_norm = torch.norm(g)
    expected = -radius * g / g_norm
    torch.testing.assert_close(step_dir[0], expected, rtol=1e-2, atol=1e-2)
    assert param_norm(step_dir) <= radius + 1e-10


def test_exact_tr_non_positive_definite(diag_obj, diag_params, diag_grad):
    # Use a Hessian with negative eigenvalue
    class NonPDCurvature(ExactHessianCalculator):
        def scaling_matrix(self, objective, params):
            H = super().scaling_matrix(objective, params)
            H[0, 0] -= 3.0  # make first eigenvalue negative
            return H

    curv = NonPDCurvature(damping=None)
    radius = 1.0
    solver = ExactTRSolver(curvature_estimator=curv, iters=20, tol=1e-10)
    step_dir = solver.optimize_model(diag_obj, diag_params, radius, diag_grad)

    assert param_norm(step_dir) <= radius + 1e-10
    assert param_dot(diag_grad, step_dir) < 0
    # The exact solver should put the step on the boundary when Hessian is not PD
    assert abs(param_norm(step_dir).item() - radius) < 1e-6


# ----------------------------------------------------------------------
# Test the factory function
# ----------------------------------------------------------------------


def test_create_trust_region_solver():
    curv = NaiveIdentityCalculator()
    solver = create_trust_region_solver("cauchy", curv)
    assert isinstance(solver, CauchyPointTRSolver)

    solver = create_trust_region_solver("dogleg", curv)
    assert isinstance(solver, DoglegTRSolver)

    solver = create_trust_region_solver("exact", curv, iters=10)
    assert isinstance(solver, ExactTRSolver)
    assert solver.iters == 10

    with pytest.raises(ValueError):
        create_trust_region_solver("unknown", curv)


# ----------------------------------------------------------------------
# Test that the model value computed by TrustRegionSolver is correct
# ----------------------------------------------------------------------


def test_model_evaluation(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    solver = CauchyPointTRSolver(curvature_estimator=exact_curvature)
    loss = scalar_obj.loss(*scalar_params)
    step = torch.tensor([0.5], dtype=torch.float64)
    # Model: m(p) = loss + g·p + 0.5 p^T H p
    g = scalar_grad[0]
    H = exact_curvature.scaling_matrix(scalar_obj, scalar_params)
    expected_model = loss + (g * step).sum() + 0.5 * step @ H @ step
    model_val = solver.model(scalar_obj, (step,), scalar_params, loss, scalar_grad)
    torch.testing.assert_close(model_val, expected_model, rtol=1e-5, atol=1e-6)
