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

class TrustRegionSolver(ABC):
    def __init__(self):
        self.cache = {}
    
    def model_computation(self, p, **kwargs):
        if p in self.cache:
            return self.cache[p]
    
    def clear_cache(self):
        self.cache = {}
        
    @abstractmethod
    def trust_region(params, step_dir, d_p_list, model_radius, eval_model):
        """ """

class CauchyPointTrustRegionSolver(TrustRegionSolver):
    def trust_region(params, step_dir, d_p_list, model_radius, eval_model):
        pass

class DoglegTrustRegionSolver(TrustRegionSolver):
    """
    Note: Not recommended for Deep learning since it underperforms on non-convex optimization.
    """

    def trust_region(params, step_dir, d_p_list, model_radius, eval_model):
        pass