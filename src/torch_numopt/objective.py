"""
Defines the objective function interface and a supervised learning implementation.

The :class:`ObjectiveFunction` abstract class provides the core abstraction for
optimization: it wraps a loss function, handles parameter storage, and manages
batched evaluation. The :class:`SupervisedLearningObjective` specializes it for
common machine learning tasks with data batching.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
import torch
from torch.func import functional_call
from .utils import param_dot, Params


class ObjectiveFunction(ABC):
    """
    Abstract base class for an objective (loss) function.

    An objective is callable and, when called, computes the loss and performs
    backpropagation to populate gradients of its parameters. This completely
    matches the closure used by pytorch optimizers, and this class can be passed
    to such optimizers as if it were a closure.

    Attributes
    ----------
    params : Params
        Tuple of parameter tensors to be optimized.
    optimizer : torch.optim.Optimizer
        The optimizer that will use this objective (used for zeroing gradients).
    batched : bool, optional
        Whether the objective supports batching (i.e., the loss is evaluated
        over sub-sets of the data). Defaults to ``False``.
    """

    def __init__(self, params: Params, optimizer, batched=False):
        self.params = tuple(params)
        self.optimizer = optimizer
        self.batched = batched

    def __call__(self) -> torch.Tensor:
        """
        Call the objective, computing the loss and its gradients.

        Returns
        -------
        torch.Tensor
            The scalar loss value.
        """

        return self.closure()

    def closure(self) -> torch.Tensor:
        """
        Compute the loss and backpropagate gradients.

        Zeroes the gradients of the optimizer, evaluates the loss, and calls
        `.backward()` on it. This is the standard closure expected by many
        PyTorch optimizers.

        Returns
        -------
        torch.Tensor
            The scalar loss value with gradients attached.
        """

        self.optimizer.zero_grad()
        loss = self.loss(*self.params, batch_idx=None)
        loss.backward()
        return loss

    @abstractmethod
    def loss(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        """
        Evaluate the objective function at the given parameters.

        Parameters
        ----------
        *params : Params
            Parameter tensors (must match the number and shape stored in
            ``self.params``).
        batch_idx : int, optional
            If the objective is batched, this index selects the batch. If
            ``None``, the full dataset is used.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """

    def residual(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        """
        Compute the residual vector (e.g., prediction-target difference).

        This method is used by Gauss-Newton and similar algorithms that
        require the residual (not just the loss). The default implementation
        raises ``NotImplementedError``.

        Parameters
        ----------
        *params : Params
            Parameter tensors.
        batch_idx : int, optional
            Batch index for batched objectives.

        Returns
        -------
        torch.Tensor
            Residual tensor (shape depends on the problem).
        """

        raise NotImplementedError("Residual calculation is not implemented.")


class SupervisedLearningObjective(ObjectiveFunction):
    """
    Objective function for supervised learning problems.

    This class wraps a PyTorch model, a loss function, and a data loader (X, y)
    to provide a standard objective. It supports mini-batch evaluation and
    L₂ weight decay.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters are to be optimized.
    loss_fn : torch.nn.Module
        Loss function (e.g., ``torch.nn.MSELoss``) that defines the criterion.
    optimizer : torch.optim.Optimizer
        Optimizer used to zero gradients.
    weight_decay : float, default=0
        Coefficient for L₂ regularization added to the loss.
    batch_size : int, optional
        If provided, the objective will be evaluated in batches of this size
        (used for memory efficiency). If ``None``, the whole dataset is used
        at once.
    """

    def __init__(self, model, loss_fn, optimizer, weight_decay=0, batch_size=None):
        super().__init__(params=model.parameters(), optimizer=optimizer, batched=batch_size is not None)
        self.model = model
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.param_keys = dict(model.named_parameters()).keys()
        self.batch_size = batch_size
        self.X = None
        self.y = None
        self.data_size = 1
        self.n_batches = None
        self.reduction = None
        if hasattr(loss_fn, "reduction"):
            self.reduction = loss_fn.reduction

    def set_data(self, x, y):
        """
        Set the training data.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        y : torch.Tensor
            Target labels.
        """

        self.X = x
        self.y = y
        if self.batch_size is None:
            self.n_batches = 1
        else:
            self.n_batches = math.ceil(len(x) / self.batch_size)
        self.data_size = x.shape[0]

    def batch_data_size(self, batch_idx):
        """
        Get the size of a specific batch (the last batch may be smaller).

        Parameters
        ----------
        batch_idx : int
            Batch index.

        Returns
        -------
        int
            Number of samples in that batch.
        """

        return min(self.batch_size, self.data_size - batch_idx * self.batch_size)

    def get_batch(self, batch_idx: int = None):
        """
        Retrieve the data slice corresponding to a batch.

        Parameters
        ----------
        batch_idx : int, optional
            Batch index. If ``None``, the full dataset is returned.

        Returns
        -------
        tuple (X, y)
            Input and target tensors for the selected batch.
        """

        if batch_idx is None:
            X = self.X
            y = self.y
        else:
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            X = self.X[batch_start:batch_end]
            y = self.y[batch_start:batch_end]

        return X, y

    def loss(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        """
        Compute the supervised loss at the given parameters.

        The loss is the sum (or mean, according to the loss function's reduction)
        of the criterion over the selected batch, plus the weight-decay term.

        Parameters
        ----------
        *params : Params
            Parameter tensors.
        batch_idx : int, optional
            Batch index; if None, use the full dataset.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """

        X, y = self.get_batch(batch_idx)

        out = functional_call(self.model, dict(zip(self.param_keys, params)), X)
        loss = self.loss_fn(out, y)
        if self.weight_decay > 0:
            loss += self.weight_decay * param_dot(params, params)

        return loss

    def residual(self, *params: Params, batch_idx: int = None) -> torch.Tensor:
        """
        Compute the residual vector (model output minus target).

        Parameters
        ----------
        *params : Params
            Parameter tensors.
        batch_idx : int, optional
            Batch index; if None, use the full dataset.

        Returns
        -------
        torch.Tensor
            Residual tensor of shape (batch_size, output_dim).
        """

        X, y = self.get_batch(batch_idx)

        out = functional_call(self.model, dict(zip(self.param_keys, params)), X)
        return out - y
