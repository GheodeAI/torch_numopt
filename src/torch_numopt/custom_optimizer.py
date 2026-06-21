""" """

from __future__ import annotations
from typing import Callable, Iterable
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from .objective import ObjectiveFunction


class CustomOptimizer(Optimizer, ABC):
    """
    Class for Optimization methods using second derivative information.
    """

    @abstractmethod
    def step(
        self,
        objective: ObjectiveFunction
    ) -> Iterable:
        """
        Method to update the parameters of the Neural Network.

        Parameters
        ----------

        """

    def update(self, loss: float):
        """
        Function to update the internal parameters of the optimization procedure.

        loss: float
            Loss of the Neural Network with the new parameters.
        """
