import torch


def torch_to_float(a: float | torch.Tensor):
    """Convert a tensor to a Python float (detaches and extracts value)."""

    if isinstance(a, torch.Tensor):
        return a.detach().item()
    else:
        return float(a)
