""" """

from __future__ import annotations
from typing import Callable, Iterable
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class CustomOptimizer(Optimizer, ABC):
    """
    Class for Optimization methods using second derivative information.
    """

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
        closure: Callable | None = None,
    ) -> Iterable:
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
        closure: Callable
            Kept for compatibility, unused.
        """

    def update(self, loss: float):
        """
        Function to update the internal parameters of the optimization procedure.

        loss: float
            Loss of the Neural Network with the new parameters.
        """
