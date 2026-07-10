"""
Trust-region methods for optimization.

Trust-region algorithms compute a step by solving a subproblem within a region
where the quadratic model is trusted. This module provides Cauchy point,
dogleg, exact (with Lagrange multiplier), and Steihaug-Toint (CG) solvers.
"""

from abc import ABC, abstractmethod
import logging
import torch

from torch_numopt.utils.param_operations import param_zero_like
from .utils import (
    param_scaled_add,
    param_norm,
    param_dot,
    param_scalar_prod,
    param_add,
    param_reshape_like,
    param_flatten,
    param_neg,
    param_diff,
    Params,
)
from .curvature_estimator import CurvatureEstimator
from .solve_system import solve_system
from .objective import ObjectiveFunction

logger = logging.getLogger(__name__)

tr_methods = {"cauchy", "dogleg", "exact"}


def create_trust_region_solver(method, curvature_estimator, solver="solve", **kwargs):
    """
    Factory function for trust-region solvers.

    Parameters
    ----------
    method : str
        One of ``"cauchy"``, ``"dogleg"``, ``"exact"``, ``"steihaug-toint"``.
    curvature_estimator : CurvatureEstimator
        Curvature estimator used to build the quadratic model.
    solver : str, default="solve"
        Linear solver for exact/Steihaug-Toint methods.
    **kwargs
        Additional arguments passed to the solver constructor.

    Returns
    -------
    TrustRegionSolver
        Instance of the requested solver.
    """

    match method:
        case "cauchy":
            trust_region_method = CauchyPointTRSolver(curvature_estimator=curvature_estimator, solver=solver)
        case "dogleg":
            trust_region_method = DoglegTRSolver(curvature_estimator=curvature_estimator, solver=solver)
        case "exact":
            print("Using exact trust region solver. Expect heavy memory use or even OOM exceptions.")
            trust_region_method = ExactTRSolver(curvature_estimator=curvature_estimator, **kwargs)
        case "steihaug-toint":
            trust_region_method = SteihaugTointTRSolver(curvature_estimator=curvature_estimator, **kwargs)
        case _:
            tr_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in tr_methods])
            last_comma_idx = tr_methods_str.rfind(",")
            tr_methods_str = tr_methods_str[:last_comma_idx] + " or" + tr_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Trust region method {method} does not exist. Try {tr_methods_str}.")

    return trust_region_method


class TrustRegionSolver(ABC):
    """
    Abstract base class for trust-region subproblem solvers.

    Subclasses must implement the ``optimize_model`` method to compute a step
    that approximately minimizes the quadratic model within a given radius.

    Parameters
    ----------
    curvature_estimator : CurvatureEstimator
        Estimator used for the quadratic model.
    solver : str, default="solve"
        Linear solver for steps that require solving a linear system.
    """

    def __init__(self, curvature_estimator: CurvatureEstimator, solver: str = "solve"):
        self.curvature_estimator = curvature_estimator
        self.solver = solver

    def model(self, objective: ObjectiveFunction, step_dir: Params, params: Params, loss: float, grad_params: Params):
        """
        Evaluate the quadratic model at a given step.

        The model is m(p) = f + gᵀp + ½ pᵀ H p.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        step_dir : Params
            Candidate step p.
        params : Params
            Current parameters.
        loss : float
            Current loss value f.
        grad_params : Params
            Current gradient g.

        Returns
        -------
        torch.Tensor
            Model value m(p).
        """

        if isinstance(step_dir, int) and step_dir == 0:
            return loss

        grad_step_dot = param_dot(grad_params, step_dir)
        model_value = loss + grad_step_dot + 0.5 * self.curvature_estimator.quadratic_form(objective, params, step_dir)

        return model_value

    @abstractmethod
    def optimize_model(self, objective: ObjectiveFunction, params: Params, radius: float, grad_params: Params):
        """
        Solve the trust-region subproblem.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Current parameters.
        radius : float
            Trust-region radius.
        grad_params : Params
            Current gradient.

        Returns
        -------
        Params
            Step direction (p) that lies within the trust region.
        """


class ExactTRSolver(TrustRegionSolver):
    """
    Exact trust-region solver using the Lagrange multiplier method.

    Solves the problem minimize m(p) subject to ||p|| ≤ Δ by finding the root
    of the secular equation. This is computationally expensive as it requires
    factorizing the matrix (H + λI).

    Parameters
    ----------
    curvature_estimator : CurvatureEstimator
        Curvature estimator.
    iters : int, default=20
        Maximum number of iterations for the root-finding.
    tol : float, default=1e-12
        Tolerance for the norm constraint.
    """

    def __init__(self, curvature_estimator: CurvatureEstimator, iters: int = 20, tol=1e-12):
        super().__init__(curvature_estimator)
        self.iters = iters
        self.tol = tol

    def optimize_model(self, objective, params, radius, grad_params):
        B = self.curvature_estimator.full_scaling_matrix(objective, params)
        grad_flat = param_flatten(grad_params)
        try:
            next_lambda = 0
            L = torch.linalg.cholesky(B)

            p = -torch.cholesky_solve(grad_flat.unsqueeze(-1), L).squeeze(-1)
            p_norm = torch.linalg.norm(p)
            if p_norm - radius <= self.tol:
                step_dir = param_reshape_like(p, params)

                # DEBUG
                if logger.isEnabledFor(logging.DEBUG):
                    gdotp = torch.dot(grad_flat, p).item()
                    model_red = -gdotp - 0.5 * self.curvature_estimator.quadratic_form(objective, params, param_reshape_like(-p, params)).item()
                    logger.debug(f"[TR] λ={next_lambda:9.4f}  ||p||={p_norm:7.5f}  gᵀp={gdotp:+8.5f}  Δm={model_red:+8.5f}")
                return step_dir

            q = torch.linalg.solve_triangular(L, p.unsqueeze(-1), upper=False).squeeze(-1)
            q_norm = torch.linalg.norm(q)

            next_lambda = (p_norm / q_norm) ** 2 * (p_norm - radius) / radius
        except torch.linalg.LinAlgError as e:
            logger.debug("Hessian matrix is non SDP.")
            next_lambda = 1e4

        eye = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
        for _ in range(self.iters):
            try:
                new_B = B + next_lambda * eye
                L = torch.linalg.cholesky(new_B)
            except torch.linalg.LinAlgError as e:
                next_lambda = max(next_lambda * 10, 1e-4)
                continue

            p = -torch.cholesky_solve(grad_flat.unsqueeze(-1), L).squeeze(-1)
            p_norm = torch.linalg.norm(p)
            if abs(p_norm - radius) <= self.tol:
                break

            q = torch.linalg.solve_triangular(L, p.unsqueeze(-1), upper=False).squeeze(-1)
            q_norm = torch.linalg.norm(q)

            next_lambda = next_lambda + (p_norm / q_norm) ** 2 * (p_norm - radius) / radius

        step_dir = param_reshape_like(p, params)

        # DEBUG
        if logger.isEnabledFor(logging.DEBUG):
            p_norm = torch.linalg.norm(p).item()
            gdotp = torch.dot(grad_flat, p).item()
            model_red = -gdotp - 0.5 * self.curvature_estimator.quadratic_form(objective, params, param_reshape_like(-p, params)).item()
            logger.debug(f"[TR] λ={next_lambda:9.4f}  ||p||={p_norm:7.5f}  gᵀp={gdotp:+8.5f}  Δm={model_red:+8.5f}")

        return step_dir


class CauchyPointTRSolver(TrustRegionSolver):
    """
    Cauchy point trust-region solver.

    The Cauchy point is the step that minimizes the quadratic model along the
    steepest descent direction within the trust region. It is cheap and ensures
    a minimum decrease.
    """

    def optimize_model(self, objective: ObjectiveFunction, params: Params, radius: float, grad_params: Params):
        eps = torch.finfo(params[0].dtype).eps

        grad_norm = param_norm(grad_params)
        g_B_g = self.curvature_estimator.quadratic_form(objective, params, grad_params)

        if g_B_g <= 0:
            tau = 1
        else:
            tau = min(grad_norm**3 / (radius * g_B_g + eps), 1)

        scaling = tau * radius / (grad_norm + eps)
        step_dir = param_scalar_prod(-scaling, grad_params)

        return step_dir


class DoglegTRSolver(TrustRegionSolver):
    """
    Dogleg trust-region solver.

    Combines the steepest descent and Newton steps to form a piecewise linear
    path. The step is the point on this path that reaches the trust-region
    boundary.
    """

    def optimize_model(self, objective: ObjectiveFunction, params: Params, radius: float, grad_params: Params):
        eps = torch.finfo(grad_params[0].dtype).eps

        grad_norm_sq = param_dot(grad_params, grad_params)
        g_B_g = self.curvature_estimator.quadratic_form(objective, params, grad_params)

        if g_B_g <= 0:
            scaling = radius / (torch.sqrt(grad_norm_sq) + eps)
            step_dir = param_scalar_prod(-scaling, grad_params)
            return step_dir

        grad_scale = grad_norm_sq / (g_B_g + eps)
        psd = param_scalar_prod(-grad_scale, grad_params)
        norm_psd = param_norm(psd)

        if norm_psd >= radius:
            step_dir = param_scalar_prod(radius / norm_psd, psd)
            return step_dir

        pgn = solve_system(self.curvature_estimator, objective, grad_params, solver=self.solver)
        pgn = param_neg(pgn)

        norm_pgn = param_norm(pgn)
        if norm_pgn <= radius:
            step_dir = pgn
            return step_dir

        a = psd
        b = param_diff(pgn, psd)
        aa = param_dot(a, a)
        bb = param_dot(b, b)
        ab = param_dot(a, b)
        c = aa - radius * radius
        t = (-ab + torch.sqrt(ab * ab - bb * c)) / (bb + eps)

        step_dir = param_scaled_add(a, b, scale=t)
        return step_dir


class SteihaugTointTRSolver(TrustRegionSolver):
    """
    Steihaug-Toint conjugate gradient trust-region solver.

    Uses the CG method to solve the trust-region subproblem, with early
    termination when the boundary is reached or negative curvature is detected.
    This is the recommended method for large-scale problems.

    Parameters
    ----------
    curvature_estimator : CurvatureEstimator
        Curvature estimator.
    max_iter : int, default=20
        Maximum CG iterations.
    atol : float, default=1e-8
        Absolute tolerance for residual norm.
    tol : float, default=1e-4
        Relative tolerance for residual norm.
    min_iter : int, default=2
        Minimum number of CG iterations before early stop.
    """

    def __init__(self, curvature_estimator: CurvatureEstimator, max_iter=20, atol=1e-8, tol=1e-4, min_iter=2):
        super().__init__(curvature_estimator)
        self.max_iter = max_iter
        self.atol = atol
        self.tol = tol
        self.min_iter = min_iter

    def optimize_model(self, objective, params, radius, grad_params):
        eps = torch.finfo(grad_params[0].dtype).eps
        rad_sq = radius * radius

        z = param_zero_like(grad_params)
        r = grad_params
        d = param_neg(r)
        r_sq_old = param_dot(r, r)

        grad_norm = torch.sqrt(r_sq_old)
        effective_tol = max(self.atol, self.tol * grad_norm)
        if grad_norm < effective_tol:
            return z

        for i in range(self.max_iter):
            Bd = self.curvature_estimator.hvp(objective, params, d)
            dBd = param_dot(d, Bd)
            z_old = z
            if dBd <= 0:
                d_sq = param_dot(d, d)
                aux1 = param_dot(d, z_old) / d_sq
                aux2 = (rad_sq - param_dot(z_old, z_old)) / d_sq
                tau = -aux1 + torch.sqrt(torch.clamp(aux1 * aux1 + aux2, min=0))
                new_p = param_scaled_add(z_old, d, scale=tau)
                return new_p

            alpha = r_sq_old / (dBd + eps)
            z = param_scaled_add(z, d, scale=alpha)
            if param_dot(z, z) >= rad_sq:
                d_sq = param_dot(d, d)
                aux1 = param_dot(d, z_old) / d_sq
                aux2 = (rad_sq - param_dot(z_old, z_old)) / d_sq
                tau = -aux1 + torch.sqrt(torch.clamp(aux1 * aux1 + aux2, min=0))
                new_p = param_scaled_add(z_old, d, scale=tau)
                return new_p

            r = param_scaled_add(r, Bd, scale=alpha)
            r_sq_new = param_dot(r, r)
            if torch.sqrt(r_sq_new) < effective_tol and i >= self.min_iter:
                return z

            beta = r_sq_new / r_sq_old
            d = param_scaled_add(param_neg(r), d, scale=beta)
            r_sq_old = r_sq_new

        return z
