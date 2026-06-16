from __future__ import annotations
from typing import Iterable, Optional
import logging
from copy import copy
import torch
from torch import nn
from functools import partial
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator
from ..utils import param_dot, param_scalar_prod, param_add

logger = logging.getLogger(__name__)


class GaussNewtonBlockApproximation(CurvatureEstimator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: Optional[int] = None,
        vectorize: bool = True,
        damping: Optional[str] = None,
        mu: float = 1e-4,
        uses_blocks: bool = True,
    ):
        super().__init__(model=model, batch_size=batch_size, ndim=2, uses_blocks=uses_blocks)
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

        scale = 1 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        def get_residuals_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return out - y

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

            h_list = None
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
                if h_list is None:
                    h_list = h_list_batch
                else:
                    h_list = param_add(h_list, h_list_batch)

                logger.info("Computed batch %d for the approximate hessian...", i)

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the approximate hessian...")
            for i, h in enumerate(h_list):
                if self.damping == "identity":
                    h_list[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_list[i] = h + self.mu * torch.diag(h.diagonal())
                else:
                    raise ValueError("Invalid damping strategy.")

        return h_list

    def _jvp(self, step_dir):
        model_params = tuple(self.model.parameters())
        step_dir = tuple(step_dir)

        def get_residuals_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return out - y

        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Jacobian vector product.")
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            _, jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, v=step_dir)
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Jacobian vector product, split in %d batches of size %d.", len(batch_start), self.batch_size)

            jac_dot_step = None
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                _, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, model_params, v=step_dir)

                if jac_dot_step is None:
                    jac_dot_step = jac_dot_step_batch
                else:
                    jac_dot_step = param_add(jac_dot_step, jac_dot_step_batch)

                logger.info("Computed batch %d for the Jacobian vector product...", i)

        return jac_dot_step

    def hvp(self, step_dir):
        model_params = tuple(self.model.parameters())
        step_dir = tuple(step_dir)

        scale = 1 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        def get_residuals_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return out - y

        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Gauss-Newton Hessian-vector product.")
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            residuals, jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, v=step_dir)
            _, hess_approx = torch.autograd.functional.vjp(get_residuals, model_params, v=jac_dot_step)
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Gauss-Newton Hessian-vector product, split in %d batches of size %d.", len(batch_start), self.batch_size)

            hess_approx = None
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                residuals, jac_dot_step_batch = torch.autograd.functional.jvp(get_residuals, model_params, v=step_dir)
                _, hess_approx_batch = torch.autograd.functional.vjp(get_residuals, model_params, v=jac_dot_step_batch)

                if hess_approx is None:
                    hess_approx = hess_approx_batch
                else:
                    hess_approx = param_add(hess_approx, hess_approx_batch)

                logger.info("Computed batch %d for the Gauss-Newton hvp...", i)

        # Damp vector
        if self.damping is not None:
            logger.info("Applying damping to the Gauss-Newton hvp...")
            if self.damping == "identity":
                hess_approx = param_add(hess_approx, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return param_scalar_prod(scale, hess_approx)

    def quadratic_form(self, d_p_list: Iterable[torch.Tensor]) -> torch.Tensor:
        scale = 1 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        Jp = self._jvp(d_p_list)
        quadratic_form = param_dot(Jp, Jp)

        # Damp vector
        if self.damping is not None:
            logger.info("Applying damping to the exact hessian...")
            if self.damping == "identity":
                quadratic_form = quadratic_form + self.mu * param_dot(d_p_list, d_p_list)
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return scale * quadratic_form
