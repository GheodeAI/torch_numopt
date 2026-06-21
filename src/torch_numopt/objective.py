from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import math
import torch
from torch.func import functional_call
from .utils import param_dot, Params

class ObjectiveFunction(ABC):
    """Objective function abstract class"""

    def __init__(self, params: Params, optimizer, batched=False):
        self.params = tuple(params)
        self.optimizer = optimizer
        self.batched = batched

    def __call__(self) -> torch.Tensor:
        return self.closure()

    def closure(self) -> torch.Tensor:
        """

        Returns
        -------
        torch.Tensor
            Loss value with gradient calculation
        """

        self.optimizer.zero_grad()
        loss = self.loss(*self.params, batch_idx=None)
        loss.backward()
        return loss

    @abstractmethod
    def loss(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
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

    def residual(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
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
    def __init__(self, model, loss_fn, optimizer, weight_decay=0, batch_size=None):
        super().__init__(params=model.parameters(), optimizer=optimizer, batched=batch_size is not None)
        self.model = model
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.param_keys = dict(model.named_parameters()).keys()
        self.batch_size = batch_size
        self.X = None
        self.y = None
        self.n_batches = None
        self.scale = 1
    
    def set_data(self, x, y):
        self.X = x
        self.y = y
        if self.batch_size is None:
            self.n_batches = 1
        else:
            self.n_batches = math.ceil(len(x) / self.batch_size)
        self.scale = 1/len(x)
    
    def get_batch(self, batch_idx: int = None):
        if batch_idx is None:
            X = self.X
            y = self.y
        else:
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            X = self.X[batch_start : batch_end]
            y = self.y[batch_start : batch_end]
        
        return X, y

    def loss(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        X, y = self.get_batch(batch_idx)

        out = functional_call(self.model, dict(zip(self.param_keys, params)), X)
        loss = self.loss_fn(out, y)
        if self.weight_decay > 0:
            loss += self.weight_decay * param_dot(params, params)
        
        return loss

    def residual(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        X, y = self.get_batch(batch_idx)

        out = functional_call(self.model, dict(zip(self.param_keys, params)), X)
        return out - y