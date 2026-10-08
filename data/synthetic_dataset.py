"""Artificial kinematics generated from random numbers; no external data input."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from config import OBS_LEN, PRED_LEN, DT, SEED
from utils.preprocessing import construct_sequence


class SyntheticScenes(Dataset):
    def __init__(self, scenes=12, seed=SEED):
        rng = np.random.default_rng(seed)
        self.scenes = []
        for _ in range(scenes):
            count = int(rng.integers(2, 7))
            positions = np.empty((count, OBS_LEN + PRED_LEN, 3), dtype=np.float32)
            velocities = np.empty_like(positions)
            for aircraft in range(count):
                heading = rng.uniform(-np.pi, np.pi)
                speed = rng.uniform(50, 130)
                turn = rng.choice([0.0, -0.001, 0.001])
                vertical = rng.choice([-3.0, 0.0, 3.0])
                positions[aircraft, 0] = [rng.uniform(-6000, 6000), rng.uniform(-6000, 6000), rng.uniform(1000, 3000)]
                for t in range(OBS_LEN + PRED_LEN):
                    angle = heading + turn * t * DT
                    velocities[aircraft, t] = [speed * np.cos(angle), speed * np.sin(angle), vertical]
                    if t > 0:
                        positions[aircraft, t] = positions[aircraft, t - 1] + velocities[aircraft, t] * DT
            features, targets = construct_sequence(positions, velocities)
            self.scenes.append((features, targets))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        return self.scenes[index]


def collate_scenes(scenes):
    observed = torch.cat([item[0] for item in scenes])
    future = torch.cat([item[1] for item in scenes])
    ids = torch.cat([torch.full((len(item[0]),), i, dtype=torch.long) for i, item in enumerate(scenes)])
    return observed, future, ids


def write_example(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("example_synthetic.npz")
    observed, future, ids = collate_scenes(SyntheticScenes(scenes=3, seed=SEED + 2))
    np.savez_compressed(path, observed_features=observed.numpy(), future_positions=future.numpy(),
                        scene_ids=ids.numpy(), dt=np.float32(DT), generator_seed=np.int64(SEED + 2))
    return path


if __name__ == "__main__":
    print(write_example())
