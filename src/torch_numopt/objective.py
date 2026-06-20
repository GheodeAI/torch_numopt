from abc import ABC, abstractmethod
from typing import Iterable
import torch
from torch.func import functional_call

class ObjectiveFunction(ABC):
    """Objective function abstract class"""

    def __call__(self, params: Iterable[torch.Tensor]) -> torch.Tensor:
        return self.loss(params=params, batch_idx=None)
    
    def pytorch_closure(self, params: Iterable[torch.Tensor], optimizer: torch.nn.Module) -> torch.Tensor:
        def wrapper():
            optimizer.zero_grad()
            loss = self.loss(params)
            loss.backward()
        return wrapper

    @abstractmethod
    def loss(self, params: Iterable[torch.Tensor], batch_idx: int = None) -> torch.Tensor:
        """Objective function.

        Parameters
        ----------
        params : Iterable[torch.Tensor]
            parameters to be optimized
        batch_idx : int, optional
            batch index, by default None

        Returns
        -------
        torch.Tensor
            Objective function value.
        """

    def residual(self, params: Iterable[torch.Tensor], batch_idx: int = None) -> torch.Tensor:
        """Residual calculation.

        Parameters
        ----------
        params : Iterable[torch.Tensor]
            parameters to be optimized
        batch_idx : int, optional
            batch index, by default None

        Returns
        -------
        torch.Tensor
            Objective function value.
        """

        raise NotImplementedError("Residual calculation is not implemented.")

class SupervisedLearningObjective(ObjectiveFunction):
    def __init__(self, model, loss_fn, x, y, batch_size=None, weight_decay=0):
        self.model = model
        self.loss_fn = loss_fn
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.param_keys = dict(model.named_parameters()).keys()
        self.params = tuple(model.parameters())
    
    def loss(self, params: Iterable[torch.Tensor], batch_idx: int = None) -> torch.Tensor:
        if batch_idx is None:
            x_batch = self.x
            y_batch = self.y
        else:
            x_batch = 0
            y_batch = 0

        out = functional_call(self.model, dict(zip(self.param_keys, params)), x_batch)
        return self.loss_fn(out, y_batch)

    def residual(self, params: Iterable[torch.Tensor], batch_idx: int = None) -> torch.Tensor:
        if batch_idx is None:
            x_batch = self.x
            y_batch = self.y
        else:
            x_batch = 0
            y_batch = 0

        out = functional_call(self.model, dict(zip(self.param_keys, params)), x_batch)
        return out - y_batch