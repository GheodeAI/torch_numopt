""" """

from abc import ABC, abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc, param_norm, param_dot, param_scalar_prod, param_sub, param_add
from .custom_optimizer import CustomOptimizer
from .curvature_estimator import CurvatureEstimator
from .solve_system import solve_system

tr_methods = {"cauchy", "dogleg"}


def create_trust_region_solver(method, curvature_estimator, solver="solve"):
    match method:
        case "cauchy":
            trust_region_method = CauchyPointTrustRegionSolver(curvature_estimator=curvature_estimator, solver=solver)
        case "dogleg":
            trust_region_method = DoglegTrustRegionSolver(curvature_estimator=curvature_estimator, solver=solver)
        case _:
            tr_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in tr_methods])
            last_comma_idx = tr_methods_str.rfind(",")
            tr_methods_str = tr_methods_str[:last_comma_idx] + " or" + tr_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Trust region method {method} does not exist. Try {tr_methods_str}.")

    return trust_region_method


class TrustRegionSolver(ABC):
    def __init__(self, curvature_estimator, solver="solve"):
        self.curvature_estimator = curvature_estimator
        self.solver = solver

    def model(self, step_dir, loss, d_p_list):
        """
        Computes a quadratic approximation of the loss function
        """

        if isinstance(step_dir, int) and step_dir == 0:
            return loss

        grad_step_dot = param_dot(d_p_list, step_dir)
        model_value = loss - grad_step_dot + 0.5 * self.curvature_estimator.quadratic_form(step_dir)

        return model_value

    @abstractmethod
    def optimize_model(self, params, radius, d_p_list):
        """ """


class CauchyPointTrustRegionSolver(TrustRegionSolver):
    def optimize_model(self, params, radius, d_p_list):
        eps = torch.finfo(d_p_list[0].dtype).eps

        d_p_norm = param_norm(d_p_list)
        g_B_g = self.curvature_estimator.quadratic_form(d_p_list)

        if g_B_g <= 0:
            tau = 1
        else:
            tau = min(d_p_norm**3 / (radius * g_B_g + eps), 1)

        scaling = tau * radius / (d_p_norm + eps)
        step_dir = param_scalar_prod(scaling, d_p_list)
        new_params = param_sub(params, step_dir)

        return new_params, step_dir


class DoglegTrustRegionSolver(TrustRegionSolver):
    def optimize_model(self, params, radius, d_p_list):
        eps = torch.finfo(d_p_list[0].dtype).eps

        B_list = self.curvature_estimator.scaling_matrix()
        d_p_norm = param_dot(d_p_list, d_p_list)
        g_B_g = self.curvature_estimator.quadratic_form(d_p_list)

        grad_scale = d_p_norm / (g_B_g + eps)
        psd = param_scalar_prod(grad_scale, d_p_list)

        norm_psd = param_norm(psd)
        if norm_psd >= radius:
            step_dir = param_scalar_prod((radius / norm_psd), psd)
            new_params = param_sub(params, step_dir)
            return new_params, step_dir

        pgn = solve_system(self.curvature_estimator, d_p_list, solver=self.solver)

        norm_pgn = param_norm(pgn)
        if norm_pgn <= radius:
            step_dir = pgn
            new_params = param_sub(params, step_dir)
            return new_params, step_dir

        a = psd
        b = param_sub(pgn, psd)

        aa = param_dot(a, a)
        bb = param_dot(b, b)
        ab = param_dot(a, b)
        c = aa - radius * radius
        t = (-ab + torch.sqrt(ab * ab - bb * c)) / (bb + eps)

        step_dir = param_add(a, param_scalar_prod(t, b))
        new_params = param_sub(params, step_dir)

        return new_params, step_dir
