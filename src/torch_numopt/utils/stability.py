""" """

import logging
import torch
import torch.linalg

logger = logging.getLogger(__name__)


def fix_stability(mat: torch.Tensor):
    """
    Procedure to adjust a matrix by adding a very small value to the diagonal to avoid numerical
    instability problems.

    Parameters
    ----------

    mat: torch.Tensor
        Ill conditioned matrix.

    Returns
    -------
    fixed_mat: torch.Tensor
        (Hopefully) Well conditioned matrix.

    """

    eps = torch.finfo(mat.dtype).eps
    return mat + torch.eye(mat.shape[0], device=mat.device) * eps


def fix_cond(mat):
    """Check condition number and apply `fix_stability` if ill-conditioned."""

    cond_number = torch.linalg.cond(mat)
    if cond_number > 1e8:
        mat = fix_stability(mat)

        if logger.isEnabledFor(logging.DEBUG):
            new_cond_number = torch.linalg.cond(mat)
            logger.debug("Numerical instability found, condition number was %g, new condition number is %g", cond_number, new_cond_number)

    return mat


def pinv_svd_trunc(mat: torch.Tensor, thresh: float = 1e-4):
    """
    Procedure to calculate the pseudoinverse of a matrix by using truncated SVD in order to maintain
    numerical stability.

    Parameters
    ----------

    mat: torch.Tensor
        Problematic matrix that we want to invert.
    thresh: float
        Threshold applied to the S matrix in the SVD procedure.

    Returns
    -------
    inverted_mat: torch.Tensor
       Pseudoinverse of the input matrix.
    """

    U, S, Vt = torch.linalg.svd(mat)

    S_tresh = S < thresh

    S_inv_trunc = 1.0 / S
    S_inv_trunc[S_tresh] = 0

    return Vt.T @ torch.diag(S_inv_trunc) @ U.T
