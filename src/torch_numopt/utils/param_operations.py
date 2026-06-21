import logging
from typing import Iterable
import torch
import torch.linalg

logger = logging.getLogger(__name__)

Params = Iterable[torch.Tensor]

def param_sizes(params: Params):
    """
    Obtains the shape of every matrix in the list of parameters provided.

    Parameters
    ----------
    params: list
        List of matrices containing a list of parameters.
    """

    return tuple(i.shape for i in params)


def param_reshape_like(params_flat: torch.Tensor, params: Params):
    """
    Reshapes a vector into a list of matrices with the same shapes as the `params` parameter.

    Parameters
    ----------
    params_flat: Tensor
        Vector with the parameters to reshape.
    params: list
        List of matrices with the desired shape.

    Returns
    -------
    reshaped_params: Tensor
    """

    result = []
    acc1 = 0
    acc2 = 0
    for p in params:
        flat_size = int(p.flatten().shape[0])
        acc2 += flat_size
        result.append(params_flat[acc1:acc2].reshape(p.shape))
        acc1 += flat_size

    return tuple(result)

def param_zero_like(params: Params):
    return tuple(torch.zeros_like(p) for p in params)

def param_flatten(params: Params):
    return torch.hstack(_param_flatten_rec(params))

def _param_flatten_rec(params: Params):
    all_params = []
    for i in params:
        if isinstance(i, torch.Tensor):
            all_params.append(i.flatten())
        else:
            all_params += param_flatten(i)

    return all_params

def param_norm(params: Params):
    return torch.sqrt(sum(torch.sum(p * p) for p in params))

def param_dot(params_a: Params, params_b: Params):
    return sum(torch.sum(p_a * p_b) for p_a, p_b in zip(params_a, params_b))

def param_scalar_prod(scalar: float, params: Params):
    return tuple(scalar * p for p in params)

def param_prod(params_a: Params, params_b: Params):
    return tuple(p_a * p_b for p_a, p_b in zip(params_a, params_b))

def param_add(params_a: Params, params_b: Params):
    return tuple(p_a + p_b for p_a, p_b in zip(params_a, params_b))

def param_scaled_add(params_a: Params, params_b: Params, scale: float):
    return tuple(p_a + scale * p_b for p_a, p_b in zip(params_a, params_b))

def param_sub(params_a: Params, params_b: Params):
    return tuple(p_a - p_b for p_a, p_b in zip(params_a, params_b))

def param_transpose(params: Params):
    return tuple(p.T for p in params)

def param_neg(params: Params):
    return tuple(-p for p in params)

def param_copy(params: Params):
    return tuple(p.clone() for p in params)

def param_is_finite(params: Params):
    return all(torch.all(torch.isfinite(p)) for p in params)
