from abc import ABC, abstractmethod
import torch
from .custom_optimizer import CustomOptimizer


class LineSearchOptimizer(CustomOptimizer, ABC):
    """
    Base class for gradient-based optimization algorithms with line search.
    """

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
        c1: float,
        c2: float,
        line_search_cond: str,
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
        c1: float
        c2: float
        line_search_cond: str

        Returns
        -------
        accepted: bool
        """

        accepted = True

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        match line_search_cond:
            case "armijo":
                accepted = new_loss <= loss + c1 * lr * dir_deriv
            case "wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
                armijo = new_loss <= loss + c1 * lr * dir_deriv
                curv_cond = new_dir_deriv >= c2 * dir_deriv
                accepted = armijo and curv_cond
            case "strong-wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
                armijo = new_loss <= loss + c1 * lr * dir_deriv
                curv_cond = abs(new_dir_deriv) <= c2 * abs(dir_deriv)
                accepted = armijo and curv_cond
            case "goldstein":
                accepted = loss + (1 - c1) * lr * dir_deriv <= new_loss <= loss + c1 * lr * dir_deriv
            case _:
                raise ValueError(f"Line search condition {line_search_cond} does not exist.")

        return accepted

    def backtrack(
        self,
        params: list,
        step_dir: list,
        grad: list,
        lr_init: float,
        eval_model: callable,
        c1: float,
        c2: float,
        tau: float,
        line_search_cond: str = "armijo",
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
        c1: float
        c2: float
        tau: float
        line_search_cond: str

        Returns
        -------
        new_params: list
        """

        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad, c1, c2, line_search_cond):
            lr *= tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            if lr <= 1e-10:
                break

        return new_params

    def bisect_search(self, params, step_dir, grad, lr_init, eval_model, c1, c2, line_search_cond="gradroot"):
        new_params = None

        match line_search_cond:
            case "gradroot":
                new_params = self.bisect_gradroot(params, step_dir, grad, lr_init, eval_model)
            case "wolfe" | "strong-wolfe" | "armijo":
                new_params = self.bisect_wolfe(params, step_dir, grad, lr_init, eval_model, c1, c2, line_search_cond)
        
        return new_params
    
    @torch.enable_grad()
    def bisect_wolfe(self, params, step_dir, grad, lr_init, eval_model, c1, c2, line_search_cond, iter_max=1000, tol=1e-5):

        lr = lr_init
        a_min = 0
        a_max = lr

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params)
        new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        for _ in range(iter_max):
            lr = 0.5*(a_max + a_min)

            if torch.abs(new_dir_deriv) < tol or a_max == a_min:
                break
            elif new_dir_deriv < 0:
                a_max = lr
            elif new_dir_deriv > 0:
                a_min = lr

            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        return new_params

    @torch.enable_grad()
    def bisect_gradroot(self, params, step_dir, grad, lr_init, eval_model, iter_max=1000, tol=1e-6):
        """ """

        lr = lr_init
        a_min = 0
        a_max = lr

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params)
        new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        for _ in range(iter_max):
            lr = 0.5*(a_max + a_min)

            if torch.abs(new_dir_deriv) < tol or a_max == a_min:
                break
            elif new_dir_deriv < 0:
                a_max = lr
            elif new_dir_deriv > 0:
                a_min = lr

            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])

        return new_params

    def apply_gradients(self, lr: float, eval_model: callable, params: list, d_p_list: list, h_list: list = None):
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

        match self.line_search_method:
            case "backtrack":
                new_params = self.backtrack(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.tau, self.line_search_cond)
            case "bisect":
                new_params = self.bisect_search(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.line_search_cond)
            case "const":
                new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

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
