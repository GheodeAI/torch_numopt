import torch
import pytest
from torch_numopt import (
    GradientDescentLS,
    ConjugateGradientLS,
    NewtonLS,
    GaussNewtonLS,
    LevenbergMarquardtLS,
)

@pytest.mark.parametrize("opt_class", [
    GradientDescentLS,
    ConjugateGradientLS,
    NewtonLS,
    GaussNewtonLS,
    LevenbergMarquardtLS,
])
def test_optimizer_on_nonlinear_problem(opt_class):
    # A simple nonlinear regression: y = sin(w*x) + noise
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 30).unsqueeze(1)
    y = torch.sin(2 * X) + 0.1 * torch.randn_like(X)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 1)
    )
    loss_fn = torch.nn.MSELoss()
    
    # Instantiate optimizer
    if opt_class == LevenbergMarquardtLS:
        optimizer = opt_class(model, lr_init=1.0, mu=0.1, line_search_method="backtrack")
    else:
        optimizer = opt_class(model, lr_init=1.0, line_search_method="backtrack")
    
    initial_loss = loss_fn(model(X), y).item()
    for epoch in range(5):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
        if isinstance(optimizer, LevenbergMarquardtLS):
            with torch.no_grad():
                new_loss = loss_fn(model(X), y)
            optimizer.update(new_loss)
    
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss