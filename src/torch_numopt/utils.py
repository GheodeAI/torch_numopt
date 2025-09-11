import torch


def param_sizes(params: list):
    """
    Obtains the shape of every matrix in the list of parameters provided.

    Parameters
    ----------
    params: list
        List of matrices containing a list of parameters.
    """

    return [i.shape for i in params]

def param_reshape_like(params_flat: torch.Tensor, params: list):
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
    
    return result

def param_flatten(params: list):
    return torch.hstack(_param_flatten_rec(params))

def _param_flatten_rec(params: list):
    all_params = []
    for i in params:
        if isinstance(i, torch.Tensor):
            all_params.append(i.flatten())
        else:
            all_params += param_flatten(i)
    
    return all_params


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

    return mat + torch.eye(mat.shape[0], device=mat.device) * torch.finfo(mat.dtype).eps


def pinv_svd_trunc(mat: torch.tensor, thresh: float = 1e-4):
    """
    Procedure to calculate the pseudoinverse of a matrix by using truncated SVD in order to mantain
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

    # max_val = torch.max(S)
    # S_tresh = S < thresh * max_val
    S_tresh = S < thresh

    S_inv_trunc = 1.0 / S
    S_inv_trunc[S_tresh] = 0

    return Vt.T @ torch.diag(S_inv_trunc) @ U.T
