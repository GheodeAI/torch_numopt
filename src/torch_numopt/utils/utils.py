import torch


def torch_to_float(a: float | torch.Tensor):
    if isinstance(a, torch.Tensor):
        return a.detach().item()
    else:
        return float(a)
