import pytest
import torch
from torch_numopt.line_search import (
    BacktrackingLineSearch,
    InterpolationLineSearch,
    BisectionLineSearch,
    create_line_search_solver,
)
from torch_numopt.objective import ObjectiveFunction
from torch_numopt.utils import Params


# ----------------------------------------------------------------------
# Helper: a simple objective class for testing
# ----------------------------------------------------------------------

class QuadraticObjective(ObjectiveFunction):
    """
    Objective: f(x) = 0.5 * x^T A x - b^T x, where A is a positive definite matrix.
    The gradient is A x - b.
    We'll use scalar or 2D parameters.
    """
    def __init__(self, A, b):
        # A and b are tensors; we'll treat parameters as a tuple of tensors.
        # For simplicity, we'll use a single parameter tensor of shape (n,)
        self.A = A
        self.b = b
        # Dummy optimizer and params (not used)
        super().__init__(params=(torch.zeros_like(b),), optimizer=None, batched=False)

    def loss(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        x = params[0]
        return 0.5 * x @ self.A @ x - self.b @ x

    def residual(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        raise NotImplementedError


# Helper to create a scalar quadratic: f(x) = 0.5 * a * x^2 - b*x
def scalar_quadratic(a, b):
    A = torch.tensor([[a]], dtype=torch.float64)
    b_t = torch.tensor([b], dtype=torch.float64)
    return QuadraticObjective(A, b_t)


# Helper to create a 2D quadratic with diagonal A
def diag_quadratic(diag_a, b):
    A = torch.diag(torch.tensor(diag_a, dtype=torch.float64))
    b_t = torch.tensor(b, dtype=torch.float64)
    return QuadraticObjective(A, b_t)


# ----------------------------------------------------------------------
# Fixtures for line search instances and parameters
# ----------------------------------------------------------------------

@pytest.fixture
def scalar_obj():
    # f(x) = 0.5 * 2 * x^2 - 3*x  => optimal x = b/a = 3/2 = 1.5
    return scalar_quadratic(a=2.0, b=3.0)


@pytest.fixture
def scalar_params(scalar_obj):
    # start at x=0.0
    return (torch.tensor([0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def scalar_grad(scalar_obj, scalar_params):
    # gradient at x=0: A*x - b = -b = -3
    return (torch.tensor([-3.0], dtype=torch.float64),)


@pytest.fixture
def scalar_step_dir(scalar_grad):
    # steepest descent direction: -grad = 3
    return (torch.tensor([3.0], dtype=torch.float64),)


# For a 2D problem
@pytest.fixture
def diag_obj():
    # A = diag(2, 1), b = [1, 2] -> optimal x = [0.5, 2]
    return diag_quadratic(diag_a=[2.0, 1.0], b=[1.0, 2.0])


@pytest.fixture
def diag_params(diag_obj):
    return (torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True),)


@pytest.fixture
def diag_grad(diag_obj, diag_params):
    # gradient at x=0: -b = [-1, -2]
    return (torch.tensor([-1.0, -2.0], dtype=torch.float64),)


@pytest.fixture
def diag_step_dir(diag_grad):
    # steepest descent: [1, 2]
    return (torch.tensor([1.0, 2.0], dtype=torch.float64),)


# ----------------------------------------------------------------------
# Tests for BacktrackingLineSearch
# ----------------------------------------------------------------------

def test_backtracking_armijo(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='armijo', c1=1e-4, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    # new_params = x + lr * step_dir = 0 + lr*3
    x_new = new_params[0].item()
    # Check that Armijo condition holds: f(x+lr*d) <= f(x) + c1*lr*grad_dot_d
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()  # -3 * 3 = -9
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    # And that lr is positive and reasonable
    assert lr > 0
    # For this quadratic, exact optimal lr = 1/a = 0.5 (since a=2) but with Armijo we may get smaller.
    # Check that it's not too far.
    assert 0.1 < lr < 1.0


def test_backtracking_wolfe(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='wolfe', c1=1e-4, c2=0.9, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    # Armijo
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    # Wolfe curvature: need gradient at new point
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * scalar_step_dir[0]).sum().item()
    assert new_dir_deriv >= ls.c2 * grad_dot_d
    # For quadratic, the exact lr that minimizes along line is 0.5, so Wolfe should find something close.
    assert 0.4 < lr < 0.6


def test_backtracking_strong_wolfe(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='strong-wolfe', c1=1e-4, c2=0.5, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    # Armijo
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    # Strong Wolfe: |new_dir_deriv| <= c2 * |grad_dot_d|
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * scalar_step_dir[0]).sum().item()
    assert abs(new_dir_deriv) <= ls.c2 * abs(grad_dot_d)
    # Should be near the exact minimizer (lr=0.5)
    assert 0.4 < lr < 0.6


def test_backtracking_goldstein(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='goldstein', c1=0.1, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    # Goldstein: loss + (1-c1)*lr*grad_dot_d <= new_loss <= loss + c1*lr*grad_dot_d
    lower = loss + (1 - ls.c1) * lr * grad_dot_d
    upper = loss + ls.c1 * lr * grad_dot_d
    assert lower <= new_loss <= upper


def test_backtracking_greedy(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='greedy', tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    # Greedy: new_loss <= loss
    assert new_loss <= loss


def test_backtracking_max_iter(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    # Set tau very small so it takes many iterations, but max_iter limits
    ls = BacktrackingLineSearch(condition='armijo', c1=1e-4, tau=0.9, max_iter=2)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1e-6, objective=scalar_obj
    )
    # It should have performed 2 iterations (or fewer if condition satisfied early)
    # Check that lr is not too small
    assert lr > 0
    # Since max_iter=2, it might not satisfy Armijo, but it still returns something.
    # We just check that it ran without errors.
    assert ls.n_iters_ <= 2


# ----------------------------------------------------------------------
# Tests for InterpolationLineSearch
# ----------------------------------------------------------------------

def test_interpolation_armijo(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = InterpolationLineSearch(condition='armijo', c1=1e-4, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    # For quadratic, the interpolation should be very good, lr close to 0.5
    assert 0.4 < lr < 0.6


def test_interpolation_wolfe(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = InterpolationLineSearch(condition='wolfe', c1=1e-4, c2=0.9, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * scalar_step_dir[0]).sum().item()
    assert new_dir_deriv >= ls.c2 * grad_dot_d


def test_interpolation_strong_wolfe(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = InterpolationLineSearch(condition='strong-wolfe', c1=1e-4, c2=0.5, tau=0.5)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    loss = scalar_obj.loss(*scalar_params)
    new_loss = scalar_obj.loss(*new_params)
    grad_dot_d = scalar_grad[0].item() * scalar_step_dir[0].item()
    assert new_loss <= loss + ls.c1 * lr * grad_dot_d
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * scalar_step_dir[0]).sum().item()
    assert abs(new_dir_deriv) <= ls.c2 * abs(grad_dot_d)


# ----------------------------------------------------------------------
# Tests for BisectionLineSearch
# ----------------------------------------------------------------------

def test_bisection_basic(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    # Bisection finds the minimum along the direction by setting derivative to zero.
    # For our quadratic, the derivative along d is zero at the exact minimizer.
    ls = BisectionLineSearch(tau=0.5, tol=1e-8, max_iter=100)
    new_params, lr = ls.line_search(
        scalar_params, scalar_step_dir, scalar_grad, lr_init=1.0, objective=scalar_obj
    )
    # The exact optimal step is 0.5
    assert abs(lr - 0.5) < 1e-6
    # Check that derivative is near zero
    new_loss = scalar_obj.loss(*new_params)
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * scalar_step_dir[0]).sum().item()
    assert abs(new_dir_deriv) < 1e-6


def test_bisection_2d(diag_obj, diag_params, diag_grad, diag_step_dir):
    # For 2D quadratic with A=diag(2,1), b=[1,2], start at 0.
    # The line search direction is the steepest descent direction d = [1,2].
    # The function along d: f(t) = 0.5 * (2*(t)^2 + 1*(2t)^2) - (1*t + 2*(2t))
    # = 0.5*(2t^2 + 4t^2) - (t + 4t) = 0.5*6t^2 -5t = 3t^2 -5t.
    # Minimum at t = 5/(6) ≈ 0.8333.
    ls = BisectionLineSearch(tol=1e-8, max_iter=100)
    new_params, lr = ls.line_search(
        diag_params, diag_step_dir, diag_grad, lr_init=1.0, objective=diag_obj
    )
    assert abs(lr - 5/6) < 1e-6
    # Check derivative near zero
    new_loss = diag_obj.loss(*new_params)
    new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False)[0]
    new_dir_deriv = (new_grad * diag_step_dir[0]).sum().item()
    assert abs(new_dir_deriv) < 1e-6


# ----------------------------------------------------------------------
# Tests for create_line_search_solver factory
# ----------------------------------------------------------------------

def test_create_line_search_solver():
    ls = create_line_search_solver('backtrack', 'armijo', c1=1e-4, c2=0.9, tau=0.5)
    assert isinstance(ls, BacktrackingLineSearch)
    assert ls.condition == 'armijo'
    assert ls.c1 == 1e-4

    ls = create_line_search_solver('interpolate', 'wolfe', c1=1e-3, c2=0.8, tau=0.3)
    assert isinstance(ls, InterpolationLineSearch)
    assert ls.condition == 'wolfe'

    ls = create_line_search_solver('bisect', 'armijo')  # condition unused for bisect
    assert isinstance(ls, BisectionLineSearch)

    with pytest.raises(ValueError):
        create_line_search_solver('unknown', 'armijo')


# ----------------------------------------------------------------------
# Test that line search works with batched objective (not required to test batched)
# but we can skip for now.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Test that accept_step works correctly
# ----------------------------------------------------------------------

def test_accept_step_armijo(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='armijo', c1=1e-4)
    # Test a step that should be accepted
    lr = 0.1  # small step
    new_params = (scalar_params[0] + lr * scalar_step_dir[0],)
    new_loss = scalar_obj.loss(*new_params)
    assert bool(ls.accept_step(scalar_params, new_params, scalar_step_dir, lr,
                          scalar_obj.loss(*scalar_params), new_loss, scalar_grad))

    # Test a step that should be rejected (too large)
    lr = 2.0
    new_params = (scalar_params[0] + lr * scalar_step_dir[0],)
    new_loss = scalar_obj.loss(*new_params)
    assert not bool(ls.accept_step(scalar_params, new_params, scalar_step_dir, lr,
                          scalar_obj.loss(*scalar_params), new_loss, scalar_grad))


def test_accept_step_goldstein(scalar_obj, scalar_params, scalar_grad, scalar_step_dir):
    ls = BacktrackingLineSearch(condition='goldstein', c1=0.1)
    # For Goldstein, we need a step that satisfies both lower and upper bounds.
    # At x=0, d=3, grad_dot_d = -9.
    # For lr=0.5: new_loss = f(1.5) = 0.5*2*2.25 - 3*1.5 = 2.25 - 4.5 = -2.25.
    # loss = 0.
    # armijo condition: -2.25 <= 0 + c1*0.5*(-9) = 0 - 0.45 = -0.45 -> true.
    # goldstein lower: loss + (1-c1)*lr*grad_dot_d = 0 + 0.9*0.5*(-9) = -4.05.
    # So -2.25 >= -4.05 -> true. So accepted.
    lr = 0.5
    new_params = (scalar_params[0] + lr * scalar_step_dir[0],)
    new_loss = scalar_obj.loss(*new_params)
    assert bool(ls.accept_step(scalar_params, new_params, scalar_step_dir, lr,
                          scalar_obj.loss(*scalar_params), new_loss, scalar_grad))

    # For lr=1.5: new_loss = f(4.5) = 4.5^2 - 3*4.5 = 20.25 - 13.5 = 6.75.
    # armijo: 6.75 <= 0 + 0.1*1.5*(-9) = -1.35 -> false, so rejected.
    lr = 1.5
    new_params = (scalar_params[0] + lr * scalar_step_dir[0],)
    new_loss = scalar_obj.loss(*new_params)
    assert not bool(ls.accept_step(scalar_params, new_params, scalar_step_dir, lr,
                                scalar_obj.loss(*scalar_params), new_loss, scalar_grad))