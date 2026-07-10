"""
Utility functions for parameter operations, stability, and type conversions.

These helpers operate on tuples of tensors (Params) and provide common
operations like addition, scaling, dot product, norm, reshaping, etc.
"""

from .param_operations import (
    param_dot,
    param_add,
    param_argnums,
    param_copy,
    param_detach,
    param_diff,
    param_flatten,
    param_is_finite,
    param_reshape_like,
    param_mult,
    param_neg,
    param_norm,
    param_numel,
    param_scalar_prod,
    param_scaled_add,
    param_sizes,
    param_transpose,
    param_zero_like,
    Params,
)

from .stability import fix_cond, fix_stability, pinv_svd_trunc

from .utils import torch_to_float

__all__ = [
    "param_dot",
    "param_add",
    "param_argnums",
    "param_copy",
    "param_detach",
    "param_diff",
    "param_flatten",
    "param_is_finite",
    "param_reshape_like",
    "param_mult",
    "param_neg",
    "param_norm",
    "param_numel",
    "param_scalar_prod",
    "param_scaled_add",
    "param_sizes",
    "param_transpose",
    "param_zero_like",
    "Params",
    "fix_cond",
    "fix_stability",
    "pinv_svd_trunc",
    "torch_to_float",
]
