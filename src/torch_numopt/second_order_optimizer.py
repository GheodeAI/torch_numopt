from __future__ import annotations
from abc import ABC
import torch
from functools import reduce
from .utils import param_reshape_like, param_flatten
from .line_search_optimizer import LineSearchOptimizer
from torch.autograd.functional import hessian
from torch.func import functional_call
from copy import copy


class SecondOrderOptimizer(LineSearchOptimizer, ABC):
    """
    Class for Optimization methods using second derivative information.

    Parameters
    ----------
    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        lr_init: float,
        lr_method: str = None,
        line_search_cond="armijo",
        line_search_method="const",
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        batch_size: int = None,
    ):
        super().__init__(
            model,
            lr_init=lr_init,
            lr_method=lr_method,
            line_search_cond=line_search_cond,
            line_search_method=line_search_method,
            c1=c1,
            c2=c2,
            tau=tau,
        )

        self.batch_size = batch_size

    def exact_hessian(self, x, y, loss_fn, vectorize=True):
        """
        Calculation of the exact hessian of the Neural network given a dataset.

        Parameters
        ----------
        x: torch.Tensor
            Input dataset for calculting the loss.
        y: torch.Tensor
            Target dataset for calculting the loss.
        loss_fn: torch.Module
            Loss function for which to calculate the hessian.
        vectorize: boolean
            Use vectorization in pytorch's implementation of the hessian calculation.
        """

        model_params = tuple(self._model.parameters())

        loss_fn = copy(loss_fn)
        is_mean = loss_fn.reduction == "mean"
        if is_mean:
            loss_fn.reduction = "sum"

        scale = 1 / len(x) if is_mean else 1

        def eval_model_batch(x, y, *input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        if self.batch_size is None or self.batch_size >= len(x):
            # Calculate hessian with every sample in the dataset
            eval_model = lambda *p: eval_model_batch(x, y, *p)
            h_list = torch.autograd.functional.hessian(eval_model, model_params, create_graph=True, vectorize=vectorize)
            h_list = [self._reshape_hessian(h_list[i][i] * scale) for i, _ in enumerate(h_list)]
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(x), self.batch_size)
            h_list = None
            for start in batch_start:
                # Prepare batch
                x_batch = x[start : start + self.batch_size]
                y_batch = y[start : start + self.batch_size]

                # Calculate hessian of the batch
                eval_model = lambda *p: eval_model_batch(x_batch, y_batch, *p)
                h_list_batch = torch.autograd.functional.hessian(eval_model, model_params, create_graph=True, vectorize=vectorize)
                h_list_batch = [self._reshape_hessian(h_list_batch[i][i]) * scale for i, _ in enumerate(h_list_batch)]

                # Agreggate result
                if h_list is None:
                    h_list = h_list_batch
                else:
                    h_list = [batch_h + prev_h for batch_h, prev_h in zip(h_list, h_list_batch)]

        return h_list

    def approx_hessian_gn(self, x, y, loss_fn, vectorize=True):
        """
        Calculation of the an approximate hessian of the Neural network given a dataset as in the Gauss-Newton algorithm.
        The approximate Hessian is calculated as the square of the Jacobian of the residual of every data point with respect to the parameters.

        Let the loss function be, for example the MSE:

        :math:`\mathcal{L}(x,y;\\theta) = \sum^{N}_{i=1} (f(x_i; \\theta) - y_i)^2 = \sum^{N}_{i=1} r_i`

        Then the Jacobian of the residuals will be the matrix:

        :math:`(J_{\\theta}[\mathcal{L}])_{i,j} = \dfrac{\partial r_i}{\partial \\theta_j}`

        Then, we will approximate the hessian as the product of the Jacobian with it's transpose, noting that the result
        will be a square matrix with size :math:`p\\times p` with :math:`p` being the number of parameters of the model:

        :math:`H_{\\theta}[\mathcal{L}] \\approx J_{\\theta}[\mathcal{L}]^{\intercal} \cdot J_{\\theta}[\mathcal{L}]`

        Parameters
        ----------
        x: torch.Tensor
            Input dataset for calculting the loss.
        y: torch.Tensor
            Target dataset for calculting the loss.
        loss_fn: torch.Module
            Loss function for which to calculate the hessian.
        vectorize: boolean
            Use vectorization in pytorch's implementation of the hessian calculation.
        """
        model_params = tuple(self._model.parameters())

        scale = 1 / len(x) if loss_fn.reduction == "mean" else 1

        residual_fn = copy(loss_fn)
        residual_fn.reduction = "none"

        def get_residuals_batch(x, y, *input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return residual_fn(out, y)

        # Calculate approximate Hessian matrix
        if self.batch_size is None or self.batch_size >= len(x):
            get_residuals = lambda *p: get_residuals_batch(x, y, *p)
            j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=vectorize)
            h_list = [None] * len(j_list)
            for j_idx, j in enumerate(j_list):
                j = j.flatten(start_dim=1)
                h_list[j_idx] = self._reshape_hessian(j.T @ j) * scale
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(x), self.batch_size)
            h_list = None
            for start in batch_start:
                # Prepare batch
                x_batch = x[start : start + self.batch_size]
                y_batch = y[start : start + self.batch_size]

                # Calculate appeoximate hessian of the batch
                get_residuals = lambda *p: get_residuals_batch(x_batch, y_batch, *p)
                j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=vectorize)
                h_list_batch = [None] * len(j_list)
                for j_idx, j in enumerate(j_list):
                    j = j.flatten(start_dim=1)
                    h_list_batch[j_idx] = self._reshape_hessian(j.T @ j) * scale

                # Agreggate result
                if h_list is None:
                    h_list = h_list_batch
                else:
                    h_list = [batch_h + prev_h for batch_h, prev_h in zip(h_list, h_list_batch)]

        return h_list
    
    def hutchinson_diagonal(self, x, y, loss_fn, vectorize=True, n_samples=1, as_matrix=False):
        model_params = tuple(self._model.parameters())
        params_flat = torch.hstack([i.flatten() for i in model_params])

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)
        
        h_diag_flat = torch.zeros_like(params_flat)
        for _ in range(n_samples):
            # Rademacher sample
            z_flat = 2*torch.bernoulli(torch.full_like(params_flat, 0.5, device=params_flat.device)) - 1
            z = tuple(param_reshape_like(z_flat, model_params))

            # Pytorch documentation recommends doing (vH)^T instead of Hv directly
            _, Hz = torch.autograd.functional.vhp(eval_model, model_params, v=z, create_graph=False)
            Hz_flat = torch.hstack([i.flatten() for i in Hz])

            h_diag_flat += z_flat*Hz_flat 
        h_diag_flat /= n_samples 
        
        h_diag = param_reshape_like(h_diag_flat, model_params)
        return h_diag

    @staticmethod
    def _reshape_hessian(hess: torch.Tensor):
        """
        Procedure to reshape a misshapen hessian matrix.
        The input is expected to be an array of size :math:`(X,Y,...,X,Y,...)` and the output will be
        a square matrix of size :math:`(X \cdot Y \cdots, X \cdot Y \cdots)`.


        Parameters
        ----------

        hess: torch.Tensor
            Misshapen hessian matrix.
        """

        if hess.dim() == 2:
            return hess

        if hess.dim() % 2 != 0:
            raise ValueError("Hessian has a weird ass shape.")

        # Divide shape in two halves, multiply each half to get total size
        new_shape = (
            reduce(lambda x, y: x * y, hess.size()[hess.dim() // 2 :]),
            reduce(lambda x, y: x * y, hess.size()[: hess.dim() // 2]),
        )

        assert new_shape[0] == new_shape[1], "Something weird happened with the hessian size"

        return hess.reshape(new_shape)
