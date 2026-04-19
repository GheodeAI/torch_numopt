import torch
import pytest
from torch_numopt.line_search import (
    BacktrackingLineSearch,
    InterpolationLineSearch,
    BisectionLineSearch,
    create_line_search_solver,
    ls_methods,
    ls_conditions,
)
from torch.func import functional_call

def eval_model_factory(model, X, y, loss_fn):
    param_keys = dict(model.named_parameters()).keys()
    def eval_model(*params):
        out = functional_call(model, dict(zip(param_keys, params)), X)
        return loss_fn(out, y)
    return eval_model

@pytest.mark.parametrize("method", ["backtrack", "interpolate", "bisect"])
@pytest.mark.parametrize("condition", ["armijo", "wolfe", "strong-wolfe", "goldstein"])
def test_line_search_methods(model_and_data, method, condition):
    model, X, y, loss_fn = model_and_data
    params = list(model.parameters())
    eval_model = eval_model_factory(model, X, y, loss_fn)
    
    # Compute gradient
    loss = eval_model(*params)
    grad = torch.autograd.grad(loss, params, create_graph=False)
    step_dir = [-g for g in grad]  # steepest descent direction
    lr_init = 1.0
    
    solver = create_line_search_solver(method, condition, max_iter=10, tol=1e-6)
    new_params, lr = solver(params, step_dir, grad, lr_init, eval_model)
    
    assert lr > 0
    assert solver.n_iters_ is not None
    # Check that new parameters produce lower loss (or at least not NaN)
    new_loss = eval_model(*new_params)
    assert torch.isfinite(new_loss)
    if condition != "greedy":  # armijo and others guarantee decrease
        assert new_loss <= loss + 1e-5  # slight tolerance

def test_line_search_accept_step_edge_cases(model_and_data):
    model, X, y, loss_fn = model_and_data
    params = list(model.parameters())
    eval_model = eval_model_factory(model, X, y, loss_fn)
    
    loss = eval_model(*params)
    grad = torch.autograd.grad(loss, params, create_graph=False)
    step_dir = [-g for g in grad]
    
    solver = BacktrackingLineSearch(condition="armijo", max_iter=10)
    # Test with NaN loss
    new_params = [p + torch.nan for p in params]
    assert not solver.accept_step(params, new_params, step_dir, 0.1, loss, torch.tensor(float('nan')), grad)
    # Test with infinite loss
    assert not solver.accept_step(params, new_params, step_dir, 0.1, loss, torch.tensor(float('inf')), grad)