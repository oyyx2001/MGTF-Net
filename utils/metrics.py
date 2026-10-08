"""All six metrics accept [aircraft,time,3] positions and return meters."""
import torch


def _error(predicted, target):
    predicted, target = torch.as_tensor(predicted), torch.as_tensor(target)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("Expected matching [N,T,3] positions in meters")
    if predicted.numel() == 0:
        raise ValueError("Empty trajectories")
    return predicted - target


def ade_3d(predicted, target):
    return torch.linalg.vector_norm(_error(predicted, target), dim=-1).mean().item()


def fde_3d(predicted, target):
    return torch.linalg.vector_norm(_error(predicted, target)[:, -1], dim=-1).mean().item()


def ade_horizontal(predicted, target):
    return torch.linalg.vector_norm(_error(predicted, target)[..., :2], dim=-1).mean().item()


def fde_horizontal(predicted, target):
    return torch.linalg.vector_norm(_error(predicted, target)[:, -1, :2], dim=-1).mean().item()


def mae_vertical(predicted, target):
    return _error(predicted, target)[..., 2].abs().mean().item()


def fve(predicted, target):
    return _error(predicted, target)[:, -1, 2].abs().mean().item()
