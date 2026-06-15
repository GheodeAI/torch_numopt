""" """

from abc import ABC, abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc, param_norm, param_dot, param_scalar_prod, param_sub, param_add
from .custom_optimizer import CustomOptimizer
from .curvature_estimator import CurvatureEstimator

tr_methods = {"cauchy", "dogleg"}


def create_trust_region_solver(method, scaling_matrix):
    match method:
        case "cauchy":
            trust_region_method = CauchyPointTrustRegionSolver(scaling_matrix=scaling_matrix)
        case "dogleg":
            trust_region_method = DoglegTrustRegionSolver(scaling_matrix=scaling_matrix)
        case _:
            tr_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in tr_methods])
            last_comma_idx = tr_methods_str.rfind(",")
            tr_methods_str = tr_methods_str[:last_comma_idx] + " or" + tr_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Trust region method {method} does not exist. Try {tr_methods_str}.")

    return trust_region_method


class TrustRegionSolver(ABC):
    def __init__(self, scaling_matrix):
        self.scaling_matrix = scaling_matrix

    def model(self, step_dir, loss, d_p_list):
        """
        Computes a quadratic approximation of the loss function
        """

        if isinstance(step_dir, int) and step_dir == 0:
            return loss

        grad_step_dot = param_dot(d_p_list, step_dir)
        model_value = loss - grad_step_dot + 0.5 * self.scaling_matrix.quadratic_form(step_dir)

        return model_value

    @abstractmethod
    def optimize_model(self, params, radius, d_p_list):
        """ """


class CauchyPointTrustRegionSolver(TrustRegionSolver):
    def optimize_model(self, params, radius, d_p_list):
        eps = torch.finfo(d_p_list[0].dtype).eps

        d_p_norm = param_norm(d_p_list)
        g_B_g = self.scaling_matrix.quadratic_form(d_p_list)

        if g_B_g <= 0:
            tau = 1
        else:
            tau = min(d_p_norm**3 / (radius * g_B_g + eps), 1)

        scaling = tau * radius / (d_p_norm + eps)
        step_dir = param_scalar_prod(scaling, d_p_list)
        new_params = param_sub(params, step_dir)

        return new_params, step_dir


class DoglegTrustRegionSolver(TrustRegionSolver):
    """
    Note: Not recommended for Deep learning since it underperforms on non-convex optimization. Added for completeness.
    """

    def __init__(self, scaling_matrix: CurvatureEstimator, solver="pinv"):
        self.scaling_matrix = scaling_matrix
        self.solver = solver

    def optimize_model(self, params, radius, d_p_list):
        eps = torch.finfo(d_p_list[0].dtype).eps

        B_list = self.scaling_matrix.scaling_matrix()
        d_p_norm = param_dot(d_p_list, d_p_list)
        g_B_g = self.scaling_matrix.quadratic_form(d_p_list)

        grad_scale = d_p_norm / (g_B_g + eps)
        psd = param_scalar_prod(grad_scale, d_p_list)

        norm_psd = param_norm(psd)
        if norm_psd >= radius:
            return param_scalar_prod((radius / norm_psd), psd)

        pgn = [None] * len(d_p_list)
        for i, (d_p, B) in enumerate(zip(d_p_list, B_list)):
            match self.solver:
                case "pinv":
                    B_inv = B.pinverse()
                    pgni = (B_inv @ -d_p.ravel()).reshape(d_p.shape)
                case "pinv-trunc":
                    B_inv = pinv_svd_trunc(B)
                    pgni = (B_inv @ -d_p.ravel()).reshape(d_p.shape)
                case "solve":
                    pgni = torch.linalg.solve(B, -d_p.ravel()).reshape(d_p.shape)
            
            pgn[i] = pgni

        norm_pgn = param_norm(pgn)
        if norm_pgn <= radius:
            return pgn
        
        a = psd
        b = param_sub(pgn, psd)

        aa = param_dot(a,a)
        bb = param_dot(b,b)
        ab = param_dot(a,b)
        c = aa - radius*radius
        t = (-ab + torch.sqrt(ab*ab - bb*c)) / (bb + eps)

        step_dir = param_add(a, param_scalar_prod(t, b))
        new_params = param_sub(params, step_dir)

        return new_params, step_dir
