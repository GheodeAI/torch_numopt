from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.func import functional_call
from .custom_optimizer import CustomOptimizer
from .utils import param_sizes

ls_conditions = ["armijo", "wolfe", "strong-wolfe", "goldstein"]
lr_init_methods = ["scaled", "BB1", "BB2", "quadratic", "lipschitz", "keep", None]
ls_methods = ["backtrack", "interpolate", "const"]


class LineSearchOptimizer(CustomOptimizer, ABC):
    """
    Base class for gradient-based optimization algorithms with line search.

    Parameters
    ----------
    model: nn.Module
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    line_search_cond: str (optional)
    line_search_method: str (optional)
    c1: float (optional)
    c2: float (optional)
    tau: float (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        lr_init: float = 1,
        lr_method: str = None,
        line_search_cond: str = "armijo",
        line_search_method: str = "const",
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
    ):
        assert lr_init > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr_init})

        self.lr_init = lr_init
        self.lr_method = lr_method
        self.line_search_cond = line_search_cond
        self.line_search_method = line_search_method
        self.c1 = c1
        self.c2 = c2
        self.tau = tau

        self.prev_lr = None
        self.prev_grad = None
        self.prev_step_dir = None
        self.prev_params = None

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

    @torch.enable_grad()
    def accept_step(
        self,
        params: list,
        new_params: list,
        step_dir: list,
        lr: float,
        loss: torch.Tensor,
        new_loss: torch.Tensor,
        grad: list,
    ):
        """
        Compute one of the stopping conditions for line search methods.

        Parameters
        ----------
        params: list
        new_params: list
        step_dir: list
        lr: float
        loss: torch.Tensor
        new_loss: torch.Tensor
        grad: list

        Returns
        -------
        accepted: bool
        """

        accepted = True

        grad_flat = torch.hstack([i.flatten() for i in grad])
        step_flat = torch.hstack([i.flatten() for i in step_dir])
        dir_deriv = grad_flat @ step_flat

        match self.line_search_cond:
            case "armijo":
                accepted = new_loss <= loss + self.c1 * lr * dir_deriv
            case "wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_grad_flat = torch.hstack([i.flatten() for i in new_grad])
                new_dir_deriv = new_grad_flat @ step_flat

                armijo = new_loss <= loss + self.c1 * lr * dir_deriv
                curv_cond = new_dir_deriv >= self.c2 * dir_deriv
                accepted = armijo and curv_cond
            case "strong-wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_grad_flat = torch.hstack([i.flatten() for i in new_grad])
                new_dir_deriv = new_grad_flat @ step_flat

                armijo = new_loss <= loss + self.c1 * lr * dir_deriv
                curv_cond = abs(new_dir_deriv) <= self.c2 * abs(dir_deriv)
                accepted = armijo and curv_cond
            case "goldstein":
                accepted = loss + (1 - self.c1) * lr * dir_deriv <= new_loss <= loss + self.c1 * lr * dir_deriv
            case _:
                ls_cond_str = ", ".join([f"'{i}'" if i is not None else "None" for i in ls_conditions])
                last_comma_idx = ls_cond_str.rfind(",")
                ls_cond_str = ls_cond_str[:last_comma_idx] + " or" + ls_cond_str[last_comma_idx + 1 :]
                raise ValueError(f"Line search condition {self.line_search_cond} does not exist. Try {ls_cond_str}.")

        return accepted

    @torch.enable_grad()
    def backtrack(
        self,
        params: list,
        step_dir: list,
        grad: list,
        lr_init: float,
        eval_model: callable,
    ):
        """
        Perform backtracking line search.

        Parameters
        ----------

        params: list
        step_dir: list
        grad: list
        lr_init: float
        eval_model: callable

        Returns
        -------
        new_params: list
        """

        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad):
            lr *= self.tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            if lr <= 1e-10:
                break

        return new_params, lr

    @torch.enable_grad()
    def interpolate_cubic(self, params: list, step_dir: list, grad: list, lr_init: float, eval_model: callable):
        """

        Parameters
        ----------

        params: list
        step_dir: list
        grad: list
        lr_init: float
        eval_model: callable
        """

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])
        eps = torch.finfo(dir_deriv.dtype).eps

        loss = eval_model(*params)
        lr_0 = -lr_init

        # Quadratic interpolation to obtain a new point
        # Calculate first interpolation point
        prev_params = tuple(p + lr_0 * p_step for p, p_step in zip(params, step_dir))
        prev_loss = eval_model(*prev_params)

        if self.accept_step(params, prev_params, step_dir, lr_0, loss, prev_loss, grad):
            return prev_params, lr_init

        # Calculate second interpolation point
        lr_1 = -0.5 * (dir_deriv * lr_0**2) / (prev_loss - loss - dir_deriv * lr_0 + eps)

        new_params = tuple(p + lr_1 * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        # Cubic interpolation with new calculated point
        while not self.accept_step(params, new_params, step_dir, lr_1, loss, new_loss, grad):
            if lr_0 == 0 or lr_1 == 0 or lr_1 == lr_0:
                break

            factor = 1 / ((lr_0 * lr_1) ** 2 * (lr_1 - lr_0) + eps)
            aux_mat = torch.Tensor([[lr_0**2, -(lr_1**2)], [-(lr_0**3), lr_1**3]], device=dir_deriv.device)
            aux_vec = torch.Tensor([new_loss - loss - dir_deriv * lr_1, prev_loss - loss - dir_deriv * lr_0], device=dir_deriv.device)
            a, b = factor * torch.matmul(aux_mat, aux_vec)

            lr_0 = lr_1
            lr_1 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * dir_deriv))) / (3 * a + eps)

            prev_loss = new_loss
            new_params = tuple(p + lr_1 * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

        return new_params, -lr_1

    # new_params = self.bisect_search(params, step_dir, d_p_list, lr_init, eval_model)
    def bisect_search(self, params, step_dir, d_p_list, lr_init, eval_model):
        new_params, lr = self.bisect(params, step_dir, lr_init, eval_model)
        return new_params, lr
    
    @torch.enable_grad()
    def bisect(self, params, step_dir, lr_init, eval_model, iter_max=1000, tol=1e-5):

        lr = lr_init
        a_min = 0
        a_max = lr

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params)
        new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        for _ in range(iter_max):
            if torch.abs(new_dir_deriv) < tol or a_max == a_min:
                break

            lr = 0.5*(a_max + a_min)

            if new_dir_deriv < 0:
                a_max = lr
            elif new_dir_deriv > 0:
                a_min = lr

            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        return new_params, lr

    def initialize_lr(self, lr: float, grad: list, step_dir: list, eval_model: callable, params: list):
        """

        Parameters
        ----------

        lr: float
        grad: list
        step_dir: list
        eval_model: callable
        params: list
        """

        if self.prev_lr is None:
            return lr

        grad_flat = torch.hstack([i.flatten() for i in grad])
        step_flat = torch.hstack([i.flatten() for i in step_dir])
        prev_grad_flat = torch.hstack([i.flatten() for i in self.prev_grad])
        prev_step_flat = torch.hstack([i.flatten() for i in self.prev_step_dir])

        new_lr = None
        eps = torch.finfo(params[0].dtype).eps
        match self.lr_method:
            case "scaled":
                new_lr = self.prev_lr_init * (prev_grad_flat @ prev_step_flat) / (grad_flat @ step_flat + eps)
            # Barzilai-Borwein
            case "BB1":
                new_lr = (prev_step_flat @ prev_step_flat) / (prev_step_flat @ prev_grad_flat + eps)
            case "BB2":
                new_lr = (prev_step_flat @ prev_grad_flat) / (prev_grad_flat @ prev_grad_flat + eps)
            case "quadratic":
                loss = eval_model(*params)
                new_lr = 2 * abs(loss - self.prev_loss) / (prev_grad_flat @ prev_step_flat + eps)
                new_lr = min(1.01 * new_lr, 1)
            case "lipschitz":
                grad_dist = torch.norm(grad_flat - prev_grad_flat)
                step_dist = torch.norm(step_flat - prev_step_flat)
                new_lr = step_dist / (grad_dist + eps)
            case "keep":
                new_lr = self.prev_lr
            case None:
                new_lr = lr
            case _:
                lr_init_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in lr_init_methods])
                last_comma_idx = lr_init_methods_str.rfind(",")
                lr_init_methods_str = lr_init_methods_str[:last_comma_idx] + " or" + lr_init_methods_str[last_comma_idx + 1 :]
                raise ValueError(f"Learning rate initialization method {self.lr_init} does not exist. Try {lr_init_methods_str}.")

        return new_lr

    def apply_gradients(self, eval_model: callable, params: list, d_p_list: list, h_list: list = None):
        """
        Updates the parameters of the network using a direction and a step length.

        Parameters
        ----------

        lr: float
        eval_model: callable
        params: list
        d_p_list: list
        h_list: list, optional
        """

        step_dir = self.get_step_direction(d_p_list, h_list)
        lr_init = self.initialize_lr(self.lr_init, d_p_list, step_dir, eval_model, params)

        match self.line_search_method:
            case "backtrack":
                new_params, lr = self.backtrack(params, step_dir, d_p_list, lr_init, eval_model)
            case "interpolate":
                new_params, lr = self.interpolate_cubic(params, step_dir, d_p_list, lr_init, eval_model)
            case "bisect":
                new_params, lr = self.bisect_search(params, step_dir, d_p_list, lr_init, eval_model)
            case "const":
                lr = lr_init
                new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            case _:
                ls_methods_str = ", ".join([f"'{i}'" if i is not None else "None" for i in ls_methods])
                last_comma_idx = ls_methods_str.rfind(",")
                ls_methods_str = ls_methods_str[:last_comma_idx] + " or" + ls_methods_str[last_comma_idx + 1 :]
                raise ValueError(f"Line search method {self.lr_init} does not exist. Try {ls_methods_str}.")

        self.prev_lr = lr
        self.prev_lr_init = lr_init
        self.prev_params = params
        self.prev_step_dir = step_dir
        self.prev_grad = d_p_list
        self.prev_loss = eval_model(*params)

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    @abstractmethod
    def get_step_direction(self, d_p_list: list, h_list: list):
        """
        Obtains the step direction used to update the network.

        Parameters
        ----------

        d_p_list: list
            List of gradients of the parameters.
        h_list: list
            List of Hessians of the parameters.

        Returns
        -------
        p: list
            New search direction
        """

    def get_scaling_matrix(self, 
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module
    ):
        """
        Obtains the step direction used to update the network.

        Parameters
        ----------

        d_p_list: list
            List of gradients of the parameters.
        h_list: list
            List of Hessians of the parameters.

        Returns
        -------
        p: list
            New search direction
        """
        
        return None
    
    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
    ):
        """
        Method to update the parameters of the Neural Network.

        Parameters
        ----------

        x: torch.Tensor
            Inputs of the Neural Network.
        y: torch.Tensor
            Targets of the Neural Network.
        loss_fn: nn.Module
            Loss function to be optimized.
        """

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        h_list = self.get_scaling_matrix(x, y, loss_fn)

        for group in self.param_groups:
            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, eval_model=eval_model)