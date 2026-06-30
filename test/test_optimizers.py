import pytest
import torch
from torch_numopt.numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from torch_numopt.curvature import NaiveIdentityCalculator, ExactHessianCalculator
from torch_numopt.objective import ObjectiveFunction
from torch_numopt.utils import param_dot, param_norm, param_add, param_scalar_prod, param_diff
from torch_numopt.line_search import LineSearchSolver
from torch_numopt.trust_region import TrustRegionSolver

# ----------------------------------------------------------------------
# Helper: Quadratic objective that uses the given parameters
# ----------------------------------------------------------------------


class QuadraticObjective(ObjectiveFunction):
    def __init__(self, A, b, params):
        self.A = A
        self.b = b
        super().__init__(params=params, optimizer=None, batched=False)

    def loss(self, *params, batch_idx=None):
        x = params[0]
        return 0.5 * x @ self.A @ x - self.b @ x

    def residual(self, *params, batch_idx=None):
        raise NotImplementedError


def make_scalar_quadratic(a, b, param):
    A = torch.tensor([[a]], dtype=torch.float64)
    b_t = torch.tensor([b], dtype=torch.float64)
    return QuadraticObjective(A, b_t, params=(param,))


def make_diag_quadratic(diag_a, b, param):
    A = torch.diag(torch.tensor(diag_a, dtype=torch.float64))
    b_t = torch.tensor(b, dtype=torch.float64)
    return QuadraticObjective(A, b_t, params=(param,))


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def scalar_params():
    return (torch.tensor([0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def scalar_obj(scalar_params):
    return make_scalar_quadratic(2.0, 3.0, scalar_params[0])


@pytest.fixture
def scalar_grad(scalar_obj):
    return (torch.tensor([-3.0], dtype=torch.float64),)


@pytest.fixture
def diag_params():
    return (torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def diag_obj(diag_params):
    return make_diag_quadratic([2.0, 1.0], [1.0, 2.0], diag_params[0])


@pytest.fixture
def diag_grad(diag_obj):
    return (torch.tensor([-1.0, -2.0], dtype=torch.float64),)


# ----------------------------------------------------------------------
# Mock curvature estimators
# ----------------------------------------------------------------------


@pytest.fixture
def exact_curvature():
    return ExactHessianCalculator(damping=None)


@pytest.fixture
def naive_curvature():
    return NaiveIdentityCalculator()


# ----------------------------------------------------------------------
# Mock line search solver
# ----------------------------------------------------------------------


class FixedStepLineSearch(LineSearchSolver):
    def __init__(self, fixed_lr=0.5):
        super().__init__()
        self.fixed_lr = fixed_lr

    def line_search(self, params, step_dir, grad_params, lr_init, objective):
        lr = self.fixed_lr
        new_params = param_add(params, param_scalar_prod(lr, step_dir))
        return new_params, lr


# ----------------------------------------------------------------------
# Mock trust region solver
# ----------------------------------------------------------------------


class FixedStepTrustRegion(TrustRegionSolver):
    def __init__(self, curvature_estimator, step_dir=None):
        super().__init__(curvature_estimator=curvature_estimator, solver="solve")
        self.step_dir = step_dir

    def optimize_model(self, objective, params, radius, grad_params):
        if self.step_dir is None:
            g = grad_params
            g_norm = param_norm(g)
            step = param_scalar_prod(-radius / g_norm, g)
        else:
            step = self.step_dir
        new_params = param_add(params, step)
        return new_params, step


# ----------------------------------------------------------------------
# Tests for NumericalOptimizer
# ----------------------------------------------------------------------


def test_initialize_lr_methods(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    opt = NumericalOptimizer(params=scalar_params, curvature_estimator=exact_curvature, lr_init=1.0, lr_method=None, fix_ascent=True)
    # Set previous values with non‑zero loss change
    prev_lr = 0.5
    prev_grad = (torch.tensor([-4.0], dtype=torch.float64),)
    prev_step_dir = (torch.tensor([2.0], dtype=torch.float64),)
    prev_loss = -2.0
    opt.prev_lr = prev_lr
    opt.prev_grad = prev_grad
    opt.prev_step_dir = prev_step_dir
    opt.prev_loss = prev_loss

    step_dir = (torch.tensor([1.0], dtype=torch.float64),)

    methods = [None, "keep", "scaled", "lipschitz", "BB1", "BB2", "quadratic"]
    for method in methods:
        opt.lr_method = method
        lr = opt.initialize_lr(lr=1.0, grad_params=scalar_grad, step_dir=step_dir, objective=scalar_obj, params=scalar_params)
        assert lr > 0, f"Method {method} returned non‑positive lr"

    with pytest.raises(ValueError):
        opt.lr_method = "invalid"
        opt.initialize_lr(1.0, scalar_grad, step_dir, scalar_obj, scalar_params)


def test_get_step_direction_returns_step(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    """Just verify that get_step_direction returns a non‑None step."""
    opt = NumericalOptimizer(params=scalar_params, curvature_estimator=exact_curvature, lr_init=1.0, fix_ascent=True)
    step = opt.get_step_direction(scalar_obj, scalar_grad)
    assert step is not None

    # Even with negative curvature, it should still return something
    class BadCurvature(ExactHessianCalculator):
        def scaling_matrix(self, objective, params):
            return -torch.eye(1, dtype=params[0].dtype)

    opt.curvature_estimator = BadCurvature(damping=None)
    step = opt.get_step_direction(scalar_obj, scalar_grad)
    assert step is not None


def test_apply_gradients_and_update(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    opt = NumericalOptimizer(params=scalar_params, curvature_estimator=exact_curvature, lr_init=1.0, lr_method="keep", fix_ascent=True)
    opt.prev_lr = 0.5
    opt.prev_step_dir = (torch.tensor([1.0], dtype=torch.float64),)
    opt.prev_grad = (torch.tensor([-2.0], dtype=torch.float64),)
    opt.prev_loss = scalar_obj.loss(*scalar_params).item()

    loss_old = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    loss_new = scalar_obj.loss(*scalar_params)
    assert loss_new < loss_old
    assert opt.curr_lr == 0.5
    assert opt.curr_loss == loss_new
    assert opt.curr_params is not None
    assert opt.curr_grad is not None

    opt.update_params()
    assert opt.prev_lr == 0.5
    assert opt.prev_loss == loss_new
    assert opt.prev_params is not None
    assert opt.prev_params[0] is not opt.curr_params[0]


def test_step_full_iteration(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    class DummyOptimizer:
        def zero_grad(self):
            pass

    scalar_obj.optimizer = DummyOptimizer()

    opt = NumericalOptimizer(params=scalar_params, curvature_estimator=exact_curvature, lr_init=1.0, lr_method="keep", fix_ascent=True)
    opt.prev_lr = 0.5
    opt.prev_step_dir = (torch.tensor([1.0], dtype=torch.float64),)
    opt.prev_grad = scalar_grad
    opt.prev_loss = scalar_obj.loss(*scalar_params).item()

    loss0 = scalar_obj.loss(*scalar_params)
    opt.step(scalar_obj)
    loss1 = scalar_obj.loss(*scalar_params)
    assert loss1 < loss0
    assert opt.prev_lr == opt.curr_lr
    assert opt.prev_loss == opt.curr_loss


# ----------------------------------------------------------------------
# Tests for LineSearchOptimizer
# ----------------------------------------------------------------------


def test_line_search_optimizer_uses_line_search(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    fixed_lr = 0.2
    ls = FixedStepLineSearch(fixed_lr=fixed_lr)
    opt = LineSearchOptimizer(params=scalar_params, curvature_estimator=exact_curvature, line_search=ls, lr_init=1.0, lr_method="keep")
    opt.prev_lr = 0.5
    opt.prev_step_dir = (torch.tensor([1.0], dtype=torch.float64),)
    opt.prev_grad = (torch.tensor([-2.0], dtype=torch.float64),)
    opt.prev_loss = scalar_obj.loss(*scalar_params).item()

    loss_old = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    loss_new = scalar_obj.loss(*scalar_params)
    assert loss_new < loss_old
    assert opt.curr_lr == fixed_lr


def test_line_search_optimizer_step_calls_line_search(scalar_obj, scalar_params, exact_curvature):
    class RecordingLineSearch(FixedStepLineSearch):
        def __init__(self):
            super().__init__(fixed_lr=0.3)
            self.called = False

        def line_search(self, params, step_dir, grad_params, lr_init, objective):
            self.called = True
            return super().line_search(params, step_dir, grad_params, lr_init, objective)

    ls = RecordingLineSearch()
    opt = LineSearchOptimizer(params=scalar_params, curvature_estimator=exact_curvature, line_search=ls, lr_init=1.0, lr_method=None)

    class DummyOpt:
        def zero_grad(self):
            pass

    scalar_obj.optimizer = DummyOpt()

    opt.step(scalar_obj)
    assert ls.called


# ----------------------------------------------------------------------
# TrustRegionOptimizer tests
# ----------------------------------------------------------------------


def test_trust_region_optimizer_increases_radius(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    """Radius should increase when rho>0.75 and step is on the boundary."""
    # Step norm = 1.0 (boundary) with radius = 1.0
    expected_step = (torch.tensor([1.0], dtype=torch.float64),)
    tr = FixedStepTrustRegion(curvature_estimator=exact_curvature, step_dir=expected_step)

    opt = TrustRegionOptimizer(params=scalar_params, trust_region=tr, radius_init=10.0, accept_tol=0.1, solver="solve")  # cap for increase
    opt.curvature_estimator = exact_curvature
    # Set previous radius to 1.0 (so current radius is 1.0)
    opt.prev_lr = 1.0

    loss0 = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    loss1 = scalar_obj.loss(*scalar_params)
    assert loss1 < loss0
    # rho=1, step norm=1, radius=1 -> increase to min(2*1, 10) = 2
    assert opt.curr_lr == 2.0


def test_trust_region_optimizer_decreases_radius(scalar_obj, scalar_params, scalar_grad, naive_curvature):
    """Radius should decrease when rho<0.25."""
    # Use a step that gives a poor model fit (large step with identity curvature)
    # With identity, model predicts a decrease but actual loss increases -> rho negative, so <0.25
    bad_step = (torch.tensor([5.0], dtype=torch.float64),)
    tr = FixedStepTrustRegion(curvature_estimator=naive_curvature, step_dir=bad_step)

    opt = TrustRegionOptimizer(params=scalar_params, trust_region=tr, radius_init=2.0, accept_tol=0.1, solver="solve")
    opt.curvature_estimator = naive_curvature
    # Set previous radius to 2.0 (same as init)
    opt.prev_lr = 2.0

    # Reset parameters to zero
    with torch.no_grad():
        scalar_params[0].zero_()

    loss0 = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    # Since rho<0.25, radius should be multiplied by 0.25: 2.0 * 0.25 = 0.5
    assert opt.curr_lr == 0.5


def test_trust_region_accepts_good_step(scalar_obj, scalar_params, scalar_grad, exact_curvature):
    """A step with rho>accept_tol should be accepted."""
    good_step = (torch.tensor([1.0], dtype=torch.float64),)  # gives decrease
    tr = FixedStepTrustRegion(curvature_estimator=exact_curvature, step_dir=good_step)
    opt = TrustRegionOptimizer(params=scalar_params, trust_region=tr, radius_init=1.0, accept_tol=0.1, solver="solve")
    opt.curvature_estimator = exact_curvature
    # Reset params to zero
    with torch.no_grad():
        scalar_params[0].zero_()

    loss0 = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    loss1 = scalar_obj.loss(*scalar_params)
    assert loss1 < loss0  # step accepted, loss decreased
    assert opt.curr_loss == loss1
    # Parameters should have moved
    assert scalar_params[0].item() != 0.0


def test_trust_region_rejects_bad_step(scalar_obj, scalar_params, scalar_grad, naive_curvature):
    """A step with rho<accept_tol should be rejected."""
    # Use a bad step that causes loss increase but model predicts decrease
    bad_step = (torch.tensor([5.0], dtype=torch.float64),)
    tr = FixedStepTrustRegion(curvature_estimator=naive_curvature, step_dir=bad_step)
    opt = TrustRegionOptimizer(params=scalar_params, trust_region=tr, radius_init=1.0, accept_tol=0.1, solver="solve")
    opt.curvature_estimator = naive_curvature
    # Reset params to zero
    with torch.no_grad():
        scalar_params[0].zero_()

    loss0 = scalar_obj.loss(*scalar_params)
    opt.apply_gradients(scalar_obj, scalar_params, scalar_grad)
    # Step rejected, parameters stay at 0
    assert scalar_params[0].item() == 0.0
    assert opt.curr_loss == loss0
