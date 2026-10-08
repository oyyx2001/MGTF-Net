"""Generic Cartesian sequence interfaces; no geographic or data-source parser."""
import torch
from config import OBS_LEN, PRED_LEN, DT


def build_model_input(positions, velocities=None):
    positions = torch.as_tensor(positions, dtype=torch.float32)
    if positions.ndim != 3 or positions.shape[1:] != (OBS_LEN, 3):
        raise ValueError("Expected observed positions [N,12,3] in meters")
    delta = torch.zeros_like(positions)
    delta[:, 1:] = positions[:, 1:] - positions[:, :-1]
    if velocities is None:
        velocities = delta / DT
        velocities[:, 0] = velocities[:, 1]
    velocities = torch.as_tensor(velocities, dtype=torch.float32, device=positions.device)
    if velocities.shape != positions.shape:
        raise ValueError("Velocity shape must match observed positions")
    return torch.cat((delta, velocities, positions), -1)


def construct_sequence(positions, velocities=None):
    positions = torch.as_tensor(positions, dtype=torch.float32)
    if positions.ndim != 3 or positions.shape[1:] != (OBS_LEN + PRED_LEN, 3):
        raise ValueError("Expected [N,32,3] sequence")
    observed_velocity = None if velocities is None else torch.as_tensor(velocities)[:, :OBS_LEN]
    return build_model_input(positions[:, :OBS_LEN], observed_velocity), positions[:, OBS_LEN:]


def build_scene_graph(scene_ids):
    ids = torch.as_tensor(scene_ids, dtype=torch.long, device="cpu")
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("scene_ids must be a nonempty vector")
    starts = torch.cat((torch.tensor([0]), torch.where(ids[1:] != ids[:-1])[0] + 1))
    ends = torch.cat((starts[1:], torch.tensor([ids.numel()])))
    if torch.unique(ids[starts]).numel() != starts.numel():
        raise ValueError("Aircraft belonging to each scene must be contiguous")
    return torch.stack((starts, ends), -1)


def to_core_inputs(observed_features, scene_ids):
    features = torch.as_tensor(observed_features, dtype=torch.float32)
    if features.ndim != 3 or features.shape[1:] != (OBS_LEN, 9):
        raise ValueError("Expected observed_features [N,12,9]")
    if len(scene_ids) != features.shape[0] or not torch.isfinite(features).all():
        raise ValueError("Invalid scene indexing or nonfinite features")
    native = features.transpose(0, 1) / 1000.0
    return native[:, :, :6], native[:, :, 6:], build_scene_graph(scene_ids)


def normalize_features(observed_features):
    """Fit only training observations, preserving the formal variance convention."""
    native = torch.as_tensor(observed_features, dtype=torch.float32).permute(0, 2, 1) / 1000.0
    rel_std, rel_mean = torch.std_mean(native[:, :6, 1:], dim=(0, 2))
    pos_std, pos_mean = torch.std_mean(native[:, 6:], dim=(0, 2))
    return {"rel_mean": rel_mean, "rel_std": rel_std.clamp_min(1e-3),
            "pos_mean": pos_mean, "pos_std": pos_std.clamp_min(1e-3)}
