from abc import ABC, abstractmethod
import torch

from torch_numopt.utils.param_operations import param_diff, param_dot, param_norm, param_scalar_prod
from .curvature_estimator import CurvatureEstimator
from .objective import ObjectiveFunction
from .utils import Params

step_init_methods = {"scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None}

def create_step_size_init(method: str, lr_init: float, curvature_estimator: CurvatureEstimator, min_lr=1e-18, max_lr=100, **kwargs):
    match method:
        case "constant":
            step_size_init = ConstantStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        case "keep":
            step_size_init = KeepStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        case "scaled":
            step_size_init = ScaledStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        case "interpolate":
            step_size_init = InterpolateStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        case "lipschitz":
            step_size_init = LipschitzStepSize(lr_init, curvature_estimator, min_lr, max_lr)
        case "BB1":
            step_size_init = BarzilaiBorweinStepSize(lr_init, curvature_estimator, min_lr, max_lr, long_step=False)
        case "BB2":
            step_size_init = BarzilaiBorweinStepSize(lr_init, curvature_estimator, min_lr, max_lr, long_step=True)
        case _:
            step_init_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in step_init_methods])
            last_comma_idx = step_init_methods_str.rfind(",")
            step_init_methods_str = step_init_methods_str[:last_comma_idx] + " or" + step_init_methods_str[last_comma_idx + 1 :]
            raise ValueError(f"Step size initialization method {method} does not exist. Try {step_init_methods_str}.")
    
    return step_size_init


class StepSizeInitializer(ABC):
    def __init__(self, lr_init: float, curvature_estimator: CurvatureEstimator, min_lr=1e-18, max_lr=100):
        assert lr_init > 0, "Learning rate must be a positive number."

        self.curvature_estimator = curvature_estimator
        self.lr_init = lr_init
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.prev_init_lr = None
        self.curvature_estimator = None
        self.prev_grad = None
        self.prev_step_dir = None
    
    def __call__(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        """
        Generates the initial step size to be used adjusting it appropriately.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Current parameters.
        grad_params : Params
            Gradient of the parameters.
        prev_grad : Params
            Gradient on the previous iteration.
        step_dir : Params
            Step direction.
        prev_step_dir : Params
            Step direction on the previous iteration
        prev_lr : float
            Previous step size

        Returns
        -------
        float
            Next step size.
        """

        if prev_lr is None:
            return self.lr_init

        new_lr = self.get_initial_step(objective, params, grad_params, prev_grad, step_dir, prev_step_dir, prev_lr, delta_loss)
        if isinstance(new_lr, torch.Tensor):
            new_lr = new_lr.item()
        return min(max(new_lr, self.min_lr), self.max_lr)

    @abstractmethod
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        """
        Generates the initial step size to be used.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Current parameters.
        grad_params : Params
            Gradient of the parameters.
        prev_grad : Params
            Gradient on the previous iteration.
        step_dir : Params
            Step direction.
        prev_step_dir : Params
            Step direction on the previous iteration
        prev_lr : float
            Previous step size

        Returns
        -------
        float
            Next step size.
        """

class ConstantStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        return self.lr_init

class KeepStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        return prev_lr

class ScaledStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        eps = torch.finfo(params[0].dtype).eps

        new_lr = self.prev_lr_init * param_dot(prev_grad, prev_step_dir) / (param_dot(grad_params, step_dir) + eps)
        self.prev_lr_init = new_lr

        return new_lr

class QuadraticStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        eps = torch.finfo(params[0].dtype).eps
        return -param_dot(grad_params, step_dir) / (self.curvature_estimator.quadratic_form(objective, params, step_dir) + eps)

class InterpolateStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        eps = torch.finfo(params[0].dtype).eps
        if delta_loss is None:
            new_lr = self.lr_init
        else:
            new_lr = 2 * delta_loss / (param_dot(prev_grad, prev_step_dir) + eps)
            new_lr = min(1.01 * new_lr, 1)
        return new_lr
    
class BarzilaiBorweinStepSize(StepSizeInitializer):
    def __init__(self, long_step=True, *args, **kwargs):
        self.use_long_step = long_step
        super().__init__(*args, **kwargs)

    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        eps = torch.finfo(params[0].dtype).eps
        s = param_scalar_prod(self.prev_lr, prev_step_dir)
        y = param_diff(grad_params, prev_grad)
        if self.use_long_step:
            return param_dot(s, y) / (param_dot(y, y) + eps)
        else:
            return param_dot(s, s) / (param_dot(s, y) + eps)

class LipschitzStepSize(StepSizeInitializer):
    def get_initial_step(
        self,
        objective: ObjectiveFunction,
        params: Params,
        grad_params: Params,
        prev_grad: Params,
        step_dir: Params,
        prev_step_dir: Params,
        prev_lr: float,
        delta_loss: float,
    ):
        eps = torch.finfo(params[0].dtype).eps
        s = param_scalar_prod(self.prev_lr, prev_step_dir)
        y = param_diff(grad_params, prev_grad)
        return param_norm(s) / (param_norm(y) + eps)