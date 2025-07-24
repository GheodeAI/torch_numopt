from __future__ import annotations
from abc import ABC
import torch
from functools import reduce
from .line_search_optimizer import LineSearchOptimizer
from torch.autograd.functional import hessian
from torch.func import functional_call
from copy import copy

class SecondOrderOptimizer(LineSearchOptimizer, ABC):
    """
    Class for Optimization methods using second derivative information.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        batch_size: int = None
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr})

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        self.batch_size = batch_size

    def exact_hessian(self, x, y, loss_fn, closure=None, vectorize=True):
        """
        Calculation of the exact hessian of the Neural network given a dataset.

        Parameters
        ----------
        x: torch.Tensor
        y: torch.Tensor
        loss_fn: torch.Loss??
        closure: callable
        vectorize: boolean
        """

        if closure is not None:
            raise NotImplementedError("This optimizer cannot handle closures.")

        model_params = tuple(self._model.parameters())

        loss_fn = copy(loss_fn)
        is_mean = loss_fn.reduction == "mean"
        if is_mean:
            loss_fn.reduction = "sum"

        scale = 1/len(x) if is_mean else 1


        def eval_model_batch(x, y, *input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        if self.batch_size is None:
            # Calculate hessian with every sample in the dataset
            eval_model = lambda *p: eval_model_batch(x, y, *p)
            h_list = torch.autograd.functional.hessian(eval_model, model_params, create_graph=True, vectorize=vectorize)
            h_list = [self._reshape_hessian(h_list[i][i]*scale) for i, _ in enumerate(h_list)]
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(x), self.batch_size)
            h_list = None
            for start in batch_start:
                # Prepare batch
                x_batch = x[start:start+self.batch_size]
                y_batch = y[start:start+self.batch_size]

                # Calculate hessian of the batch
                eval_model = lambda *p: eval_model_batch(x_batch, y_batch, *p)
                h_list_batch = torch.autograd.functional.hessian(eval_model, model_params, create_graph=True, vectorize=vectorize)
                h_list_batch = [self._reshape_hessian(h_list_batch[i][i])*scale for i, _ in enumerate(h_list_batch)]

                # Agreggate result
                if h_list is None:
                    h_list = h_list_batch
                else:
                    h_list = [batch_h + prev_h for batch_h, prev_h in zip(h_list, h_list_batch)]
        
        return h_list
    
    def approx_hessian_gn(self, x, y, loss_fn, closure=None, vectorize=True):
        if closure is not None:
            raise NotImplementedError("This optimizer cannot handle closures.")

        model_params = tuple(self._model.parameters())

        scale = 1/len(x) if loss_fn.reduction == "mean" else 1

        residual_fn = copy(loss_fn)
        residual_fn.reduction = "none"

        def get_residuals_batch(x, y, *input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return residual_fn(out, y)

        # Calculate approximate Hessian matrix
        if self.batch_size is None:
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
                x_batch = x[start:start+self.batch_size]
                y_batch = y[start:start+self.batch_size]

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



    @staticmethod
    def _reshape_hessian(hess: torch.Tensor):
        """
        Procedure to reshape a misshapen hessian matrix.

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
