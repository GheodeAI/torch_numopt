""" """

from abc import ABC, abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc
from .custom_optimizer import CustomOptimizer
from .scaling_matrix_calculator import ScalingMatrixCalculator

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
    
    def model(self, step_dir, loss, d_p_list, eval_model):
        """
        Computes a quadratic approximation of the loss function
        """

        if isinstance(step_dir, int) and step_dir == 0:
            return loss
        
        grad_step_dot = sum(torch.sum(p*p_step) for p, p_step in zip(d_p_list, step_dir))
        model_value = loss - grad_step_dot + 0.5 * self.scaling_matrix.hvp(step_dir)
        
        return model_value
        
    @abstractmethod
    def optimize_model(self, params, radius, d_p_list, eval_model):
        """ """

class CauchyPointTrustRegionSolver(TrustRegionSolver):
    def optimize_model(self, params, radius, d_p_list, eval_model):
        eps = torch.finfo(d_p_list[0].dtype).eps

        d_p_norm = torch.sqrt(sum(torch.sum(p**2) for p in d_p_list))
        g_B_g = self.scaling_matrix.hvp(eval_model, d_p_list)
        if g_B_g <= 0:
            tau = 1
        else:
            tau = min(d_p_norm**3/(radius * g_B_g + eps), 1)
        
        scaling = tau * radius / (d_p_norm + eps)
        step_dir = tuple(scaling * p_step for p, p_step in zip(params, d_p_list))
        new_params = tuple(p - p_step for p, p_step in zip(params, step_dir))

        return new_params, step_dir


class DoglegTrustRegionSolver(TrustRegionSolver):
    """
    Note: Not recommended for Deep learning since it underperforms on non-convex optimization.
    """

    def optimize_model(self, params, radius, d_p_list, eval_model):
        pass