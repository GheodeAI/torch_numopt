import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.func import functional_call
from torch_numopt.objective import ObjectiveFunction, SupervisedLearningObjective


# ---------- Fixtures ----------
@pytest.fixture
def simple_model():
    """A simple linear model: y = x @ W + b (no activation)."""
    return nn.Linear(2, 1, bias=True)


@pytest.fixture
def optimizer(simple_model):
    """An optimizer for the model parameters."""
    return SGD(simple_model.parameters(), lr=0.1)


@pytest.fixture
def mse_loss():
    return nn.MSELoss(reduction="mean")


@pytest.fixture
def sample_data():
    """Small synthetic dataset: 10 samples, 2 features, 1 target."""
    x = torch.randn(10, 2)
    # True weights: W = [[1.0], [2.0]], b = 0.5 -> y = x@W + b
    W_true = torch.tensor([[1.0], [2.0]])
    b_true = torch.tensor([0.5])
    y = x @ W_true + b_true + 0.1 * torch.randn(10, 1)  # add noise
    return x, y


# ---------- Tests for ObjectiveFunction (abstract base class) ----------
def test_objective_closure_calls_zero_grad_and_backward():
    """Check that closure() zeroes gradients, computes loss, and calls backward."""

    class DummyObjective(ObjectiveFunction):
        def loss(self, *params, batch_idx=None):
            return sum(p.sum() for p in params)

        def residual(self, *params, batch_idx=None):
            raise NotImplementedError

    params = (torch.tensor([1.0, 2.0], requires_grad=True),)

    class MockOptimizer:
        def __init__(self, params):
            self.params = params
            self.zero_grad_called = False

        def zero_grad(self):
            self.zero_grad_called = True

    opt = MockOptimizer(params)
    obj = DummyObjective(params, opt, batched=False)

    loss = obj.closure()

    assert opt.zero_grad_called is True
    assert torch.isclose(loss, torch.tensor(3.0))
    assert params[0].grad is not None
    assert torch.allclose(params[0].grad, torch.ones_like(params[0]))


def test_objective_abstract_methods():
    """Ensure ObjectiveFunction cannot be instantiated and residual raises NotImplementedError."""
    with pytest.raises(TypeError):
        ObjectiveFunction((), None)

    class Dummy(ObjectiveFunction):
        def loss(self, *params, batch_idx=None):
            return torch.tensor(0.0)

    dummy = Dummy((), None)
    with pytest.raises(NotImplementedError):
        dummy.residual()


# ---------- Tests for SupervisedLearningObjective ----------
# ---------- Initialization & attributes ----------
def test_supervised_initialization_stores_attributes(simple_model, mse_loss, optimizer):
    """Check that __init__ sets model, loss_fn, weight_decay, batch_size, batched flag, etc."""
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, weight_decay=0.01, batch_size=4)
    assert obj.model is simple_model
    assert obj.loss_fn is mse_loss
    assert obj.weight_decay == 0.01
    assert obj.batch_size == 4
    assert obj.batched is True
    assert obj.param_keys == dict(simple_model.named_parameters()).keys()
    assert obj.reduction == mse_loss.reduction

    obj2 = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=None)
    assert obj2.batched is False
    assert obj2.batch_size is None


# ---------- Data handling (set_data, get_batch) ----------
def test_supervised_set_data(simple_model, mse_loss, optimizer, sample_data):
    """Test set_data stores data and computes correct number of batches."""
    x, y = sample_data
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=None)
    obj.set_data(x, y)
    assert obj.X is x
    assert obj.y is y
    assert obj.data_size == x.shape[0]
    assert obj.n_batches == 1

    batch_size = 3
    obj2 = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=batch_size)
    obj2.set_data(x, y)
    assert obj2.n_batches == (len(x) + batch_size - 1) // batch_size  # ceil division


def test_supervised_get_batch(simple_model, mse_loss, optimizer, sample_data):
    """Test get_batch returns correct slices for batch indices or full data."""
    x, y = sample_data

    # Without batching
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=None)
    obj.set_data(x, y)
    X_full, y_full = obj.get_batch(None)
    assert torch.equal(X_full, x)
    assert torch.equal(y_full, y)

    # With batching
    batch_size = 3
    obj2 = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=batch_size)
    obj2.set_data(x, y)
    for i in range(obj2.n_batches):
        X_batch, y_batch = obj2.get_batch(i)
        start = i * batch_size
        end = min((i + 1) * batch_size, len(x))
        assert torch.equal(X_batch, x[start:end])
        assert torch.equal(y_batch, y[start:end])


# ---------- Loss computation ----------
def test_supervised_loss_no_weight_decay(simple_model, mse_loss, optimizer, sample_data):
    """Loss without weight decay equals MSE on the full dataset."""
    x, y = sample_data
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, weight_decay=0.0)
    obj.set_data(x, y)

    params = tuple(simple_model.parameters())
    loss_val = obj.loss(*params)

    out = simple_model(x)
    expected_loss = mse_loss(out, y)
    assert torch.isclose(loss_val, expected_loss)


def test_supervised_loss_with_weight_decay(simple_model, mse_loss, optimizer, sample_data):
    """Loss includes weight decay term: 0.5 * weight_decay * ||params||^2."""
    weight_decay = 0.1
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, weight_decay=weight_decay)
    obj.set_data(*sample_data)

    params = tuple(simple_model.parameters())
    loss_val = obj.loss(*params)

    out = simple_model(sample_data[0])
    base_loss = mse_loss(out, sample_data[1])
    reg = weight_decay * sum(torch.sum(p * p) for p in params)
    expected_loss = base_loss + reg
    assert torch.isclose(loss_val, expected_loss)


def test_supervised_loss_uses_correct_params(simple_model, mse_loss, optimizer, sample_data):
    """Ensure loss uses the passed parameters via functional_call, not the model's internal state."""
    x, y = sample_data
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer)
    obj.set_data(x, y)

    new_params = tuple(p.clone() + 1.0 for p in simple_model.parameters())
    loss_new = obj.loss(*new_params)

    param_dict = dict(zip(obj.param_keys, new_params))
    out = functional_call(simple_model, param_dict, x)
    expected_loss = mse_loss(out, y)
    assert torch.isclose(loss_new, expected_loss)


def test_supervised_loss_batched(simple_model, mse_loss, optimizer, sample_data):
    """Loss with batching computes loss only on the specified batch."""
    batch_size = 3
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=batch_size)
    obj.set_data(*sample_data)
    params = tuple(simple_model.parameters())

    for i in range(obj.n_batches):
        loss_batch = obj.loss(*params, batch_idx=i)
        X_batch, y_batch = obj.get_batch(i)
        out = simple_model(X_batch)
        expected = mse_loss(out, y_batch)
        assert torch.isclose(loss_batch, expected)


# ---------- Residual computation ----------
def test_supervised_residual(simple_model, mse_loss, optimizer, sample_data):
    """Residual returns out - y for the full dataset."""
    x, y = sample_data
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer)
    obj.set_data(x, y)
    params = tuple(simple_model.parameters())

    res = obj.residual(*params)
    out = simple_model(x)
    expected = out - y
    assert torch.allclose(res, expected)


def test_supervised_residual_batched(simple_model, mse_loss, optimizer, sample_data):
    """Residual works correctly with batching."""
    batch_size = 3
    obj = SupervisedLearningObjective(simple_model, mse_loss, optimizer, batch_size=batch_size)
    obj.set_data(*sample_data)
    params = tuple(simple_model.parameters())

    for i in range(obj.n_batches):
        res_batch = obj.residual(*params, batch_idx=i)
        X_batch, y_batch = obj.get_batch(i)
        out = simple_model(X_batch)
        expected = out - y_batch
        assert torch.allclose(res_batch, expected)
