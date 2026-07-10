""" """

import logging
from typing import Tuple
import torch

logger = logging.getLogger(__name__)

Params = Tuple[torch.Tensor]


def param_zero_like(params: Params):
    """
    Create a tuple of zero tensors with the same shapes as the input parameters.

    Parameters
    ----------
    params : Params
        Iterable of tensors providing the shapes.

    Returns
    -------
    tuple of torch.Tensor
        A tuple of tensors, each filled with zeros and with the same shape and device
        as the corresponding input tensor.
    """
    return tuple(torch.zeros_like(p) for p in params)


def param_add(params_a: Params, params_b: Params):
    """
    Element-wise addition of two parameter groups.

    Parameters
    ----------
    params_a : Params
        First iterable of tensors.
    params_b : Params
        Second iterable of tensors, must have the same structure as `params_a`.

    Returns
    -------
    tuple of torch.Tensor
        A new tuple where each element is the sum of the corresponding elements of `params_a` and `params_b`.
    """
    return tuple(p_a + p_b for p_a, p_b in zip(params_a, params_b))


def param_scalar_prod(scalar: float, params: Params):
    """
    Multiplies every tensor in a parameter group by a scalar.

    Parameters
    ----------
    scalar : float
        The scalar multiplier.
    params : Params
        Iterable of tensors to scale.

    Returns
    -------
    tuple of torch.Tensor
        New tuple containing each tensor multiplied by `scalar`.
    """
    return tuple(scalar * p for p in params)


def param_dot(params_a: Params, params_b: Params):
    """
    Computes the dot product between two parameter groups treated as a single flat vector.

    Parameters
    ----------
    params_a : Params
        First iterable of tensors.
    params_b : Params
        Second iterable of tensors, same shapes as `params_a`.

    Returns
    -------
    torch.Tensor
        Scalar tensor equal to sum_{p_a, p_b} (p_a * p_b).sum().
    """
    return sum(torch.sum(p_a * p_b) for p_a, p_b in zip(params_a, params_b))


def param_diff(params_a: Params, params_b: Params):
    """
    Element-wise subtraction of two parameter groups.

    Parameters
    ----------
    params_a : Params
        First iterable of tensors.
    params_b : Params
        Second iterable of tensors, same structure.

    Returns
    -------
    tuple of torch.Tensor
        New tuple where each element is `params_a[i] - params_b[i]`.
    """
    return tuple(p_a - p_b for p_a, p_b in zip(params_a, params_b))


def param_scaled_add(params_a: Params, params_b: Params, scale: float):
    """
    Computes `params_a + scale * params_b` element-wise.

    Parameters
    ----------
    params_a : Params
        Base parameter group.
    params_b : Params
        Parameter group to be scaled and added.
    scale : float
        Scaling factor for `params_b`.

    Returns
    -------
    tuple of torch.Tensor
        New tuple with the result of the scaled addition.
    """
    return tuple(p_a + scale * p_b for p_a, p_b in zip(params_a, params_b))


def param_norm(params: Params):
    """
    Euclidean (L2) norm of a parameter group treated as a flat vector.

    Parameters
    ----------
    params : Params
        Iterable of tensors.

    Returns
    -------
    torch.Tensor
        Scalar tensor equal to sqrt(sum_{p} (p * p).sum()).
    """
    return torch.sqrt(sum(torch.sum(p * p) for p in params))


def param_mult(params_a: Params, params_b: Params):
    """
    Element-wise (Hadamard) product of two parameter groups.

    Parameters
    ----------
    params_a : Params
        First iterable of tensors.
    params_b : Params
        Second iterable of tensors, same shapes.

    Returns
    -------
    tuple of torch.Tensor
        New tuple where each element is `params_a[i] * params_b[i]`.
    """
    return tuple(p_a * p_b for p_a, p_b in zip(params_a, params_b))


def param_transpose(params: Params):
    """
    Transposes every 2-dimensional tensor in the parameter group.

    Parameters
    ----------
    params : Params
        Iterable of tensors; only 2D tensors are transposed, others are left unchanged.

    Returns
    -------
    tuple of torch.Tensor
        New tuple with transposed tensors where applicable.
    """
    return tuple(p.mT for p in params)


def param_neg(params: Params):
    """
    Negates every tensor in a parameter group.

    Parameters
    ----------
    params : Params
        Iterable of tensors.

    Returns
    -------
    tuple of torch.Tensor
        New tuple with each element negated (`-p`).
    """
    return tuple(-p for p in params)


def param_copy(params: Params):
    """
    Creates a deep copy of a parameter group (each tensor cloned).

    Parameters
    ----------
    params : Params
        Iterable of tensors to copy.

    Returns
    -------
    tuple of torch.Tensor
        New tuple containing detached clones of the original tensors.
    """
    return tuple(p.clone() for p in params)


def param_is_finite(params: Params):
    """
    Checks that all elements in a parameter group are finite.

    Parameters
    ----------
    params : Params
        Iterable of tensors to inspect.

    Returns
    -------
    bool
        True if every element of every tensor is finite, False otherwise.
    """
    return all(torch.all(torch.isfinite(p)) for p in params)


def param_sizes(params: Params):
    """
    Obtains the shape of every matrix in the parameters provided.

    Parameters
    ----------
    params: Params
        Sequence of matrices containing a sequence of parameters.
    """

    return tuple(i.shape for i in params)


def param_reshape_like(params_flat: torch.Tensor, params: Params):
    """
    Reshapes a vector into a sequence of matrices with the same shapes as the `params` parameter.

    Parameters
    ----------
    params_flat: Tensor
        Vector with the parameters to reshape.
    params: Params
        Sequence of matrices with the desired shape.

    Returns
    -------
    reshaped_params: Tensor
    """

    assert len(params_flat) == param_numel(params), "Size mismatch"

    result = []
    acc1 = 0
    acc2 = 0
    for p in params:
        flat_size = int(p.flatten().shape[0])
        acc2 += flat_size
        result.append(params_flat[acc1:acc2].reshape(p.shape))
        acc1 += flat_size

    return tuple(result)


def param_flatten(params: Params):
    """
    Flattens an entire parameter group into a single 1-dimensional tensor.

    Parameters
    ----------
    params : Params
        Iterable of tensors (possibly nested). Each tensor is flattened and concatenated.

    Returns
    -------
    torch.Tensor
        1-D tensor containing all parameter values concatenated in order.
    """
    if len(params) == 0:
        return torch.empty(0)
    return torch.cat([p.flatten() for p in params])


def param_numel(params: Params) -> int:
    """
    Returns the total amount of parameters.

    Parameters
    ----------
    params : Params
        Iterable of tensors.

    Returns
    -------
    int
        Count of the total number of parameters.
    """

    return sum(p.numel() for p in params)


def param_detach(params: Params) -> Params:
    """Detach (and clone) all tensors from the computation graph."""

    return tuple(p.detach().clone() for p in params)


def param_argnums(params: Params) -> tuple:
    """Return a tuple of indices (0, 1, ..., len(params)-1) for the parameter groups.

    This is a convenience function used to pass `argnums` to `torch.func`
    when computing Hessians or Jacobians.
    """

    return tuple(range(len(params)))
