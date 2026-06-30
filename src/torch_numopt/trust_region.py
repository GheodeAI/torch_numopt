""" """

from abc import ABC, abstractmethod
import logging
import torch
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
    match method:
        case "cauchy":
            trust_region_method = CauchyPointTRSolver(curvature_estimator=curvature_estimator, solver=solver)
        case "dogleg":
            trust_region_method = DoglegTRSolver(curvature_estimator=curvature_estimator, solver=solver)
        case "exact":
            print("Using exact trust region solver. Expect heavy memory use or even OOM exceptions.")
            trust_region_method = ExactTRSolver(curvature_estimator=curvature_estimator, **kwargs)
        case _:
            tr_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in tr_methods])
            last_comma_idx = tr_methods_str.rfind(",")
            tr_methods_str = tr_methods_str[:last_comma_idx] + " or" + tr_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Trust region method {method} does not exist. Try {tr_methods_str}.")

    return trust_region_method


class TrustRegionSolver(ABC):
    def __init__(self, curvature_estimator: CurvatureEstimator, solver: str = "solve"):
        self.curvature_estimator = curvature_estimator
        self.solver = solver

    def model(self, objective: ObjectiveFunction, step_dir: Params, params: Params, loss: float, grad_params: Params):
        """
        Computes a quadratic approximation of the loss function
        """

        if isinstance(step_dir, int) and step_dir == 0:
            return loss

        grad_step_dot = param_dot(grad_params, step_dir)
        model_value = loss + grad_step_dot + 0.5 * self.curvature_estimator.quadratic_form(objective, params, step_dir)

        return model_value

    @abstractmethod
    def optimize_model(self, objective: ObjectiveFunction, params: Params, radius: float, grad_params: Params):
        """ """


class ExactTRSolver(TrustRegionSolver):
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
                new_params = param_add(params, step_dir)

                # DEBUG
                if logger.isEnabledFor(logging.DEBUG):
                    gdotp = torch.dot(grad_flat, p).item()
                    model_red = -gdotp - 0.5 * self.curvature_estimator.quadratic_form(objective, params, param_reshape_like(-p, params)).item()
                    logger.debug(f"[TR] λ={next_lambda:9.4f}  ||p||={p_norm:7.5f}  gᵀp={gdotp:+8.5f}  Δm={model_red:+8.5f}")
                return new_params, step_dir

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
        new_params = param_add(params, step_dir)

        # DEBUG
        if logger.isEnabledFor(logging.DEBUG):
            p_norm = torch.linalg.norm(p).item()
            gdotp = torch.dot(grad_flat, p).item()
            model_red = -gdotp - 0.5 * self.curvature_estimator.quadratic_form(objective, params, param_reshape_like(-p, params)).item()
            logger.debug(f"[TR] λ={next_lambda:9.4f}  ||p||={p_norm:7.5f}  gᵀp={gdotp:+8.5f}  Δm={model_red:+8.5f}")

        return new_params, step_dir


class CauchyPointTRSolver(TrustRegionSolver):
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
        new_params = param_add(params, step_dir)

        return new_params, step_dir


class DoglegTRSolver(TrustRegionSolver):
    def optimize_model(self, objective: ObjectiveFunction, params: Params, radius: float, grad_params: Params):
        eps = torch.finfo(grad_params[0].dtype).eps

        grad_norm_sq = param_dot(grad_params, grad_params)
        g_B_g = self.curvature_estimator.quadratic_form(objective, params, grad_params)

        if g_B_g <= 0:
            scaling = radius / (torch.sqrt(grad_norm_sq) + eps)
            step_dir = param_scalar_prod(-scaling, grad_params)
            new_params = param_add(params, step_dir)
            return new_params, step_dir

        grad_scale = grad_norm_sq / (g_B_g + eps)
        psd = param_scalar_prod(-grad_scale, grad_params)
        norm_psd = param_norm(psd)

        if norm_psd >= radius:
            step_dir = param_scalar_prod(radius / norm_psd, psd)
            new_params = param_add(params, step_dir)
            return new_params, step_dir

        pgn = solve_system(self.curvature_estimator, objective, grad_params, solver=self.solver)
        pgn = param_neg(pgn)

        norm_pgn = param_norm(pgn)
        if norm_pgn <= radius:
            step_dir = pgn
            new_params = param_add(params, step_dir)
            return new_params, step_dir

        a = psd
        b = param_diff(pgn, psd)
        aa = param_dot(a, a)
        bb = param_dot(b, b)
        ab = param_dot(a, b)
        c = aa - radius * radius
        t = (-ab + torch.sqrt(ab * ab - bb * c)) / (bb + eps)

        step_dir = param_scaled_add(a, b, scale=t)
        new_params = param_add(params, step_dir)

        return new_params, step_dir

# class Seihaug