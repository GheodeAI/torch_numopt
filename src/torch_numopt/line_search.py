"""
Line-search algorithms for step-length determination.

This module provides various line-search strategies: backtracking, interpolation,
and bisection. Each solver implements a specific method and can be combined with
different stopping conditions (Armijo, Wolfe, Goldstein, etc.).
"""

from abc import ABC, abstractmethod
import logging
import torch
from .objective import ObjectiveFunction
from .utils import param_scaled_add, param_dot, param_is_finite, param_neg, torch_to_float, Params

logger = logging.getLogger(__name__)

ls_methods = {"backtrack", "interpolate", "bisect"}
ls_conditions = {"greedy", "armijo", "wolfe", "strong-wolfe", "goldstein"}


def create_line_search_solver(method, condition, c1=1e-4, c2=0.9, tau=0.1, max_iter=20, tol=1e-8):
    """
    Factory function to instantiate a line-search solver.

    Parameters
    ----------
    method : str
        One of ``"backtrack"``, ``"interpolate"``, ``"bisect"``.
    condition : str
        Stopping condition: ``"greedy"``, ``"armijo"``, ``"wolfe"``,
        ``"strong-wolfe"``, ``"goldstein"``.
    c1 : float, default=1e-4
        Sufficient decrease parameter (Armijo).
    c2 : float, default=0.9
        Curvature condition parameter (Wolfe).
    tau : float, default=0.1
        Step-size reduction factor for backtracking.
    max_iter : int, default=20
        Maximum number of iterations.
    tol : float, default=1e-8
        Tolerance for stopping (e.g., minimum step size).

    Returns
    -------
    LineSearchSolver
        Instance of the requested solver.
    """
    match method:
        case "backtrack":
            line_search = BacktrackingLineSearch(condition=condition, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol)
        case "interpolate":
            line_search = InterpolationLineSearch(condition=condition, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol)
        case "bisect":
            line_search = BisectionLineSearch(condition=condition, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol)
        case _:
            ls_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in ls_methods])
            last_comma_idx = ls_methods_str.rfind(",")
            ls_methods_str = ls_methods_str[:last_comma_idx] + " or" + ls_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Line search method {method} does not exist. Try {ls_methods_str}.")

    return line_search


class LineSearchSolver(ABC):
    """
    Abstract base class for line-search solvers.

    Subclasses must implement the ``line_search`` method to find an acceptable
    step length.

    Parameters
    ----------
    condition : str, default="armijo"
        Stopping condition (see above).
    c1 : float, default=1e-4
        Sufficient decrease parameter.
    c2 : float, default=0.9
        Curvature condition parameter.
    tau : float, default=0.1
        Step reduction factor.
    max_iter : int, default=20
        Maximum iterations.
    tol : float, default=1e-8
        Tolerance (e.g., minimum step).
    """

    def __init__(
        self,
        condition: str = "armijo",
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
    ):
        self.condition = condition
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol

        # Debug parameters
        self.n_iters_ = None
        self.new_lr_ = None

    @torch.enable_grad()
    def accept_step(
        self,
        params: Params,
        new_params: Params,
        step_dir: Params,
        lr: float,
        loss: torch.Tensor,
        new_loss: torch.Tensor,
        grad_params: Params,
    ):
        """
        Check if the current step satisfies the chosen stopping condition.

        Parameters
        ----------
        params : Params
            Current parameters.
        new_params : Params
            Candidate parameters at step length `lr`.
        step_dir : Params
            Search direction.
        lr : float
            Step length.
        loss : torch.Tensor
            Loss at `params`.
        new_loss : torch.Tensor
            Loss at `new_params`.
        grad_params : Params
            Gradient at `params`.

        Returns
        -------
        bool
            ``True`` if the step is acceptable.
        """

        if not torch.isfinite(new_loss).all():
            return False

        elif not torch.isfinite(loss).all():
            return True

        dir_deriv = param_dot(grad_params, step_dir)
        if not torch.isfinite(dir_deriv).all():
            return False

        accepted = True
        match self.condition:
            case "greedy":
                accepted = new_loss <= loss
            case "armijo":
                accepted = new_loss <= loss + self.c1 * lr * dir_deriv
            case "wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=False)
                if not param_is_finite(new_grad):
                    return False

                new_dir_deriv = param_dot(new_grad, step_dir)

                armijo = new_loss <= loss + self.c1 * lr * dir_deriv
                curv_cond = new_dir_deriv >= self.c2 * dir_deriv
                accepted = armijo and curv_cond
            case "strong-wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=False)
                if not param_is_finite(new_grad):
                    return False
                new_dir_deriv = param_dot(new_grad, step_dir)

                armijo = new_loss <= loss + self.c1 * lr * dir_deriv
                curv_cond = abs(new_dir_deriv) <= self.c2 * abs(dir_deriv)
                accepted = armijo and curv_cond
            case "goldstein":
                accepted = loss + (1 - self.c1) * lr * dir_deriv <= new_loss <= loss + self.c1 * lr * dir_deriv
            case _:
                ls_cond_str = ", ".join([f"'{i}'" if i is not None else "None" for i in ls_conditions])
                last_comma_idx = ls_cond_str.rfind(",")
                ls_cond_str = ls_cond_str[:last_comma_idx] + " or" + ls_cond_str[last_comma_idx + 1 :]
                raise ValueError(f"Line search condition {self.condition} does not exist. Try {ls_cond_str}.")

        logger.info("Step was %s.", "accepted" if accepted else "rejected")

        return accepted

    @abstractmethod
    @torch.enable_grad()
    def find_step_size(
        self,
        params: Params,
        step_dir: Params,
        grad_params: Params,
        lr_init: float,
        objective: ObjectiveFunction,
    ):
        """
        Perform the line search.

        Parameters
        ----------
        params : Params
            Current parameters.
        step_dir : Params
            Search direction.
        grad_params : Params
            Gradient at current point.
        lr_init : float
            Initial step length.
        objective : ObjectiveFunction
            Objective function.

        Returns
        -------
        tuple (new_params, lr)
            The updated parameters and the chosen step length.
        """


class BacktrackingLineSearch(LineSearchSolver):
    """
    Backtracking line search with step reduction.

    Starting from `lr_init`, the step size is repeatedly multiplied by `tau`
    until the chosen condition is satisfied or `max_iter` is reached.
    """

    @torch.enable_grad()
    def find_step_size(
        self,
        params: Params,
        step_dir: Params,
        grad_params: Params,
        lr_init: float,
        objective: ObjectiveFunction,
    ):
        lr = lr_init

        loss = objective.loss(*params)

        new_params = param_scaled_add(params, step_dir, lr)
        new_loss = objective.loss(*new_params)

        logger.info("Starting backtracking line search with initial guess of %g with loss of %g.", lr, new_loss)

        n_iters = 0
        while n_iters < self.max_iter and not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad_params) and lr >= self.tol:
            lr *= self.tau

            # Evaluate model with new lr
            new_params = param_scaled_add(params, step_dir, lr)
            new_loss = objective.loss(*new_params)

            logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, lr, new_loss)

            n_iters += 1

        if n_iters >= self.max_iter:
            logger.debug("Exceeded the maximum number of line search iterations.")

        logger.info("Settled into lr = %g.", lr)

        self.n_iters_ = n_iters
        self.new_lr_ = float(lr)

        return new_params, lr


class InterpolationLineSearch(LineSearchSolver):
    """
    Line search using quadratic/cubic interpolation.

    Uses the function and derivative values at two points to fit a polynomial
    and estimate the optimum step length.
    """

    @torch.enable_grad()
    def find_step_size(
        self,
        params: Params,
        step_dir: Params,
        grad_params: Params,
        lr_init: float,
        objective: ObjectiveFunction,
    ):
        dir_deriv = param_dot(grad_params, step_dir)
        device = dir_deriv.device
        dtype = dir_deriv.dtype

        eps = torch.finfo(dtype).eps

        loss = objective.loss(*params)

        # Respect sign convention in "Numerical Optimization" by Noceadal J., which assumes maximization.
        # The sign is reverted at the end of the function.
        lr_0 = torch.tensor(lr_init, device=device, dtype=dtype)

        # Quadratic interpolation to obtain a new point
        # Calculate first interpolation point
        prev_params = param_scaled_add(params, step_dir, scale=lr_0)
        prev_loss = objective.loss(*prev_params)

        if self.accept_step(params, prev_params, step_dir, lr_init, loss, prev_loss, grad_params):
            return prev_params, lr_init

        # Calculate second interpolation point
        lr_1 = -0.5 * (dir_deriv * lr_0**2) / (prev_loss - loss - dir_deriv * lr_0 + eps)

        new_params = param_scaled_add(params, step_dir, scale=lr_1)
        new_loss = objective.loss(*new_params)

        logger.info("Starting interpolation line search with initial guess of %g with loss of %g.", -lr_1, new_loss)

        # Cubic interpolation with new calculated point
        n_iters = 0
        while (
            n_iters < self.max_iter
            and not self.accept_step(params, new_params, step_dir, -lr_1, loss, new_loss, grad_params)
            and not torch.isclose(lr_1, lr_0, rtol=1e-8, atol=1e-10)
            and lr_1 >= self.tol
        ):
            factor = 1 / ((lr_0 * lr_1) ** 2 * (lr_1 - lr_0) + eps)
            aux_mat = torch.tensor([[lr_0**2, -(lr_1**2)], [-(lr_0**3), lr_1**3]], device=device, dtype=dtype)
            aux_vec = torch.tensor(
                [
                    new_loss - loss - dir_deriv * lr_1,
                    prev_loss - loss - dir_deriv * lr_0,
                ],
                device=device,
                dtype=dtype,
            )
            a, b = factor * torch.matmul(aux_mat, aux_vec)

            lr_0 = lr_1
            lr_1 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * dir_deriv))) / (3 * a + eps)

            prev_loss = new_loss
            new_params = param_scaled_add(params, step_dir, scale=lr_1)
            new_loss = objective.loss(*new_params)

            logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, -lr_1, new_loss)

            n_iters += 1

        if n_iters >= self.max_iter:
            logger.debug("Exceeded the maximum number of line search iterations.")

        logger.info("Settled into lr = %g.", lr_1)

        self.n_iters_ = n_iters
        self.new_lr_ = float(-lr_1.detach().item())

        return new_params, lr_1


class BisectionLineSearch(LineSearchSolver):
    """
    Bisection (binary search) line search.

    It maintains an interval containing the optimal step and narrows it
    by checking the derivative sign.
    """

    @torch.enable_grad()
    def find_step_size(self, params, step_dir, grad_params, lr_init, objective):
        loss = objective.loss(*params)
        a_min = 0
        a_max = lr_init

        lr = a_max
        new_params = param_scaled_add(params, step_dir, scale=lr)
        new_loss = objective.loss(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=True)
        new_dir_deriv = param_dot(new_grad, step_dir)

        if self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad_params):
            if logger.isEnabledFor(logging.INFO):
                logger.info("Accepted initial lr = %g.", lr)
            return new_params, lr

        if new_dir_deriv > 0:
            a_max = lr
        else:
            a_min = lr

        n_iters = 0
        for _ in range(self.max_iter):
            lr = 0.5 * (a_max + a_min)
            new_params = param_scaled_add(params, step_dir, scale=lr)
            new_loss = objective.loss(*new_params)
            new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=True)
            new_dir_deriv = param_dot(new_grad, step_dir)

            if self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad_params):
                break

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, torch_to_float(lr), torch_to_float(new_loss))

            if new_dir_deriv > 0:
                a_max = lr
            else:
                a_min = lr

            if a_max - a_min <= self.tol:
                break

            n_iters += 1

        if logger.isEnabledFor(logging.INFO):
            logger.info("Settled into lr = %g.", lr)

        self.n_iters_ = n_iters
        self.new_lr_ = float(lr)

        return new_params, lr
