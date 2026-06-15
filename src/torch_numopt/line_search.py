""" """

from abc import ABC, abstractmethod
from typing import Callable
import logging
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc
from .custom_optimizer import CustomOptimizer

logger = logging.getLogger(__name__)

ls_methods = {"backtrack", "interpolate", "bisect"}
ls_conditions = {"greedy", "armijo", "wolfe", "strong-wolfe", "goldstein"}

def create_line_search_solver(method, condition, c1=1e-4, c2=0.9, tau=0.1, max_iter=20, tol=1e-8):
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
        params: list,
        new_params: list,
        step_dir: list,
        lr: float,
        loss: torch.Tensor,
        new_loss: torch.Tensor,
        grad: list,
    ):
        """
        Compute one of the stopping conditions for line search methods.

        Parameters
        ----------
        params: list
        new_params: list
        step_dir: list
        lr: float
        loss: torch.Tensor
        new_loss: torch.Tensor
        grad: list

        Returns
        -------
        accepted: bool
        """

        if not torch.isfinite(new_loss).all():
            return False
        elif not torch.isfinite(loss).all():
            return True

        grad_flat = torch.hstack([i.ravel() for i in grad])
        step_flat = torch.hstack([i.ravel() for i in step_dir])
        dir_deriv = grad_flat @ step_flat
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
                new_grad_flat = torch.hstack([i.ravel() for i in new_grad])
                if not torch.isfinite(new_grad_flat).all():
                    return False

                new_dir_deriv = new_grad_flat @ step_flat

                armijo = new_loss <= loss + self.c1 * lr * dir_deriv
                curv_cond = new_dir_deriv >= self.c2 * dir_deriv
                accepted = armijo and curv_cond
            case "strong-wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=False)
                new_grad_flat = torch.hstack([i.ravel() for i in new_grad])
                if not torch.isfinite(new_grad_flat).all():
                    return False
                new_dir_deriv = new_grad_flat @ step_flat

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
        
        logger.info("Step was %s.", 'accepted' if accepted else 'rejected')

        return accepted

    def __call__(self, params, step_dir, grad, lr_init, eval_model):
        return self.line_search(params, step_dir, grad, lr_init, eval_model)

    @abstractmethod
    @torch.enable_grad()
    def line_search(
        self,
        params: list,
        step_dir: list,
        grad: list,
        lr_init: float,
        eval_model: Callable,
    ):
        """"""


class BacktrackingLineSearch(LineSearchSolver):
    @torch.enable_grad()
    def line_search(
        self,
        params: list,
        step_dir: list,
        grad: list,
        lr_init: float,
        eval_model: Callable,
    ):
        """
        Perform backtracking line search.

        Parameters
        ----------

        params: list
        step_dir: list
        grad: list
        lr_init: float
        eval_model: Callable

        Returns
        -------
        new_params: list
        """

        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        logger.info("Starting backtracking line search with initial guess of %g with loss of %g.", lr, new_loss)

        n_iters = 0
        while (
            n_iters < self.max_iter 
            and not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad)
            and lr >= self.tol
        ):
            lr *= self.tau


            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, lr, new_loss)

            n_iters += 1

        if n_iters >= self.max_iter:
            logger.debug("Exceeded the maximum number of line search iterations.")

        logger.info("Settled into lr = %g.", lr)
        
        self.n_iters_ = n_iters
        self.new_lr_ = float(lr)

        return new_params, lr


class InterpolationLineSearch(LineSearchSolver):
    @torch.enable_grad()
    def line_search(
        self,
        params: list,
        step_dir: list,
        grad: list,
        lr_init: float,
        eval_model: Callable,
    ):
        """

        Parameters
        ----------

        params: list
        step_dir: list
        grad: list
        lr_init: float
        eval_model: Callable
        """

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])
        eps = torch.finfo(dir_deriv.dtype).eps

        loss = eval_model(*params)
        
        # Respect sign convention in "Numerical Optimization" by Noceadal J., which assumes maximization.
        # The sign is reverted at the end of the function.
        lr_0 = -lr_init

        # Quadratic interpolation to obtain a new point
        # Calculate first interpolation point
        prev_params = tuple(p + lr_0 * p_step for p, p_step in zip(params, step_dir))
        prev_loss = eval_model(*prev_params)


        if self.accept_step(params, prev_params, step_dir, lr_init, loss, prev_loss, grad):
            return prev_params, lr_init

        # Calculate second interpolation point
        lr_1 = -0.5 * (dir_deriv * lr_0**2) / (prev_loss - loss - dir_deriv * lr_0 + eps)

        new_params = tuple(p + lr_1 * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        logger.info("Starting interpolation line search with initial guess of %g with loss of %g.", -lr_1, new_loss)

        # Cubic interpolation with new calculated point
        n_iters = 0
        while (
            n_iters < self.max_iter 
            and not self.accept_step(params, new_params, step_dir, -lr_1, loss, new_loss, grad)
            and torch.isclose(lr_1, lr_0, rtol=1e-8, atol=1e-10)
            and lr_1 >= self.tol
        ):
            factor = 1 / ((lr_0 * lr_1) ** 2 * (lr_1 - lr_0) + eps)
            aux_mat = torch.tensor([[lr_0**2, -(lr_1**2)], [-(lr_0**3), lr_1**3]], device=dir_deriv.device)
            aux_vec = torch.tensor(
                [
                    new_loss - loss - dir_deriv * lr_1,
                    prev_loss - loss - dir_deriv * lr_0,
                ],
                device=dir_deriv.device,
            )
            a, b = factor * torch.matmul(aux_mat, aux_vec)

            lr_0 = lr_1
            lr_1 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * dir_deriv))) / (3 * a + eps)

            prev_loss = new_loss
            new_params = tuple(p + lr_1 * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, -lr_1, new_loss)

            n_iters += 1
        
        if n_iters >= self.max_iter:
            logger.debug("Exceeded the maximum number of line search iterations.")

        logger.info("Settled into lr = %g.", -lr_1)

        self.n_iters_ = n_iters
        self.new_lr_ = float(-lr_1.detach().item())

        return new_params, -lr_1


class BisectionLineSearch(LineSearchSolver):
    @torch.enable_grad()
    def line_search(self, params, step_dir, d_p_list, lr_init, eval_model):
        lr = lr_init
        a_min = 0
        a_max = lr

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        new_loss = eval_model(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=False)
        new_dir_deriv = sum(torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir))

        logger.info("Starting bisection line search with initial guess of %g with loss of %g.", lr, new_loss)

        n_iters = 0
        while n_iters < self.max_iter and torch.abs(new_dir_deriv) >= self.tol and a_max != a_min:
            lr = 0.5 * (a_max + a_min)

            if new_dir_deriv < 0:
                a_max = lr
            elif new_dir_deriv > 0:
                a_min = lr

            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            logger.debug("Iteration %d, new guess is %g which yielded a loss of %g.", n_iters, lr, new_loss)

            new_grad = torch.autograd.grad(new_loss, new_params, create_graph=False, retain_graph=False)
            new_dir_deriv = sum(torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir))
            n_iters += 1

        if n_iters >= self.max_iter:
            logger.debug("Exceeded the maximum number of line search iterations.")

        logger.info("Settled into lr = %g.", lr)

        self.n_iters_ = n_iters
        self.new_lr_ = float(lr)

        return new_params, lr
