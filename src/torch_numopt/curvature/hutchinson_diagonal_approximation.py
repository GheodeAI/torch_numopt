from __future__ import annotations
from typing import Iterable
import logging
import torch
from torch import nn
from ..utils import param_reshape_like, param_scalar_prod, param_prod, param_add, param_zero_like
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator

logger = logging.getLogger(__name__)


class HutchinsonDiagonalApproximation(CurvatureEstimator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = None,
        n_samples: int = 1,
    ):
        super().__init__(model=model, batch_size=batch_size, ndim=1, uses_blocks=False)
        self.n_samples = n_samples

    def scaling_matrix(self) -> Iterable:
        model_params = tuple(self.model.parameters())
        params_flat = torch.hstack([i.ravel() for i in model_params])

        def eval_model(*input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), self.x_)
            return self.loss_fn_(out, self.y_)

        h_diag = param_zero_like(model_params)
        logger.info("Computing diagonal Hutchinson approximation of the hessian with %d samples.", self.n_samples)
        for i in range(self.n_samples):
            # Rademacher sample
            z_flat = 2 * torch.randint(0, 2, size=params_flat.size(), device=params_flat.device, dtype=torch.uint8).float() - 1
            z = param_reshape_like(z_flat, model_params)

            # Pytorch documentation recommends doing (vH)^T instead of Hv directly
            _, Hz = torch.autograd.functional.vhp(eval_model, model_params, v=z, create_graph=False)

            h_diag = param_add(h_diag, param_prod(z, Hz))

            logger.info("Calculated approximation for random sample number %d...", i)

        h_diag = param_scalar_prod(1/self.n_samples, h_diag)

        return h_diag

    def hvp(self, step_dir):
        diag_hessian = self.scaling_matrix()
        return param_prod(step_dir, diag_hessian)

    def quadratic_form(self, d_p_list: Iterable[torch.Tensor]) -> torch.Tensor:

        scaling_matrix_dot_grad = self.hvp(d_p_list)
        quadratic_form = param_scalar_prod(d_p_list, scaling_matrix_dot_grad)

        return quadratic_form
