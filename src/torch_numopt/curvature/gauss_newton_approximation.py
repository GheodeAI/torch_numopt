from __future__ import annotations
from typing import Iterable, Optional
import logging
from copy import copy
import torch
from torch import nn
from functools import reduce, partial
from ..utils import param_reshape_like
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator

logger = logging.getLogger(__name__)


class GaussNewtonBlockApproximation(CurvatureEstimator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: Optional[int] = None,
        vectorize: bool = True,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.vectorize = vectorize
        self.damping = damping
        self.mu = mu

    def scaling_matrix(self) -> Iterable:
        r"""
        Calculation of the an approximate hessian of the Neural network given a dataset as in the Gauss-Newton algorithm.
        The approximate Hessian is calculated as the square of the Jacobian of the residual of every data point with respect to the parameters.

        Let the loss function be, for example the MSE:

        :math:`\mathcal{L}(x,y;\theta) = \sum^{N}_{i=1} (f(x_i; \theta) - y_i)^2 = \sum^{N}_{i=1} r_i`

        Then the Jacobian of the residuals will be the matrix:

        :math:`(J_{\theta}[\mathcal{L}])_{i,j} = \dfrac{\partial r_i}{\partial \theta_j}`

        Then, we will approximate the hessian as the product of the Jacobian with it's transpose, noting that the result
        will be a square matrix with size :math:`p\\times p` with :math:`p` being the number of parameters of the model:

        :math:`H_{\theta}[\mathcal{L}] \approx J_{\theta}[\mathcal{L}]^{\intercal} \cdot J_{\theta}[\mathcal{L}]`

        Parameters
        ----------
        x: torch.Tensor
            Input dataset for calculating the loss.
        y: torch.Tensor
            Target dataset for calculating the loss.
        loss_fn: torch.Module
            Loss function for which to calculate the hessian.
        vectorize: boolean
            Use vectorization in pytorch's implementation of the hessian calculation.
        """
        model_params = tuple(self.model.parameters())

        scale = 2 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        residual_fn = copy(self.loss_fn_)
        residual_fn.reduction = "none"

        def get_residuals_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return residual_fn(out, y)

        # Calculate approximate Hessian matrix
        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Gauss-Newton approximate Hessian matrix.")
            # get_residuals = lambda *p: get_residuals_batch(x, y, *p)
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=self.vectorize)
            h_list = [None] * len(j_list)
            for j_idx, j in enumerate(j_list):
                j = j.view(j.shape[0], -1)
                h_list[j_idx] = self._reshape_hessian(j.T @ j) * scale
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", len(batch_start), self.batch_size)

            h_list = []
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=self.vectorize)
                h_list_batch = [None] * len(j_list)
                for j_idx, j in enumerate(j_list):
                    j = j.view(j.shape[0], -1)
                    h_list_batch[j_idx] = self._reshape_hessian(j.T @ j) * scale

                # Aggregate result
                if h_list == []:
                    h_list = h_list_batch
                else:
                    h_list = [batch_h + prev_h for batch_h, prev_h in zip(h_list, h_list_batch)]

                logger.info("Computed batch %d for the approximate hessian...", i)

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the approximate hessian...")
            for i, h in enumerate(h_list):
                if self.damping == "identity":
                    h_list[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_list[i] = h + self.mu * h.diagonal()
                else:
                    raise ValueError("Invalid damping strategy.")

        return h_list

    def hvp(self, step_dir):
        model_params = tuple(self.model.parameters())

        scale = 2 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        residual_fn = copy(self.loss_fn_)
        residual_fn.reduction = "none"

        def get_residuals_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return residual_fn(out, y)

        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Gauss-Newton approximate Hessian matrix.")
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, tangents=step_dir)
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", len(batch_start), self.batch_size)

            h_list = []
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, tangents=step_dir)

                hess_dot_step += hess_dot_step_batch
                logger.info("Computed batch %d for the Gauss-newton approximate hessian...", i)

        return hess_dot_step * scale

    def quadratic_form(d_p_list: Iterable[torch.Tensor]) -> torch.Tensor:

        scaling_matrix_dot_grad = self.hvp(d_p_list)
        quadratic_form = sum(torch.sum(hvi * hvi) for hvi in zip(scaling_matrix_dot_grad))  # p Jt J p = |Jp|^2

        return quadratic_form
