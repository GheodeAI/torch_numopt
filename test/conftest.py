import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Small synthetic regression problem
@pytest.fixture
def linear_data():
    torch.manual_seed(42)
    X = torch.randn(20, 2)
    true_w = torch.tensor([1.5, -2.0])
    true_b = torch.tensor(0.5)
    y = X @ true_w + true_b + 0.1 * torch.randn(20)
    return X, y

@pytest.fixture
def simple_model():
    return nn.Linear(2, 1)

@pytest.fixture
def mse_loss():
    return nn.MSELoss()

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model_and_data(linear_data, simple_model, mse_loss, device):
    X, y = linear_data
    model = simple_model.to(device)
    X, y = X.to(device), y.to(device)
    return model, X, y, mse_loss