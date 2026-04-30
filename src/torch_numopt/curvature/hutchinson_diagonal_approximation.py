from ..curvature_estimator import CurvatureEstimator

class HutchinsonDiagonalApproximation(CurvatureEstimator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = None,
        n_samples: int = 1,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.n_samples = n_samples

    def scaling_matrix(self) -> Iterable:
        model_params = tuple(self.model.parameters())
        params_flat = torch.hstack([i.ravel() for i in model_params])

        def eval_model(*input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), self.x_)
            return self.loss_fn_(out, self.y_)

        h_diag_flat = torch.zeros_like(params_flat)
        logger.info("Computing diagonal Hutchinson approximation of the hessian with %d samples.", self.n_samples)
        for i in range(self.n_samples):
            # Rademacher sample
            z_flat = 2 * torch.bernoulli(torch.full_like(params_flat, 0.5, device=params_flat.device)) - 1
            z = tuple(param_reshape_like(z_flat, model_params))

            # Pytorch documentation recommends doing (vH)^T instead of Hv directly
            _, Hz = torch.autograd.functional.vhp(eval_model, model_params, v=z, create_graph=False)
            Hz_flat = torch.hstack([i.ravel() for i in Hz])

            h_diag_flat += z_flat * Hz_flat

            logger.info("Calculated approximation for random sample number %d...", i)
        h_diag_flat /= self.n_samples

        h_diag = param_reshape_like(h_diag_flat, model_params)
        return h_diag

    def hvp(self, step_dir):
        diag_hessian = self.scaling_matrix
        return tuple(p * h for p, h in zip(step_dir, diag_hessian))

    def quadratic_form(d_p_list: Iterable[torch.Tensor]) -> torch.Tensor:

        scaling_matrix_dot_grad = self.hvp(d_p_list)
        quadratic_form = sum(torch.sum(vi * hvi) for vi, hvi in zip(d_p_list, scaling_matrix_dot_grad))

        return quadratic_form