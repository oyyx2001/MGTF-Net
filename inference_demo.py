"""Evaluate the artificial example; values are not scientific benchmark results."""
from pathlib import Path
import numpy as np
import torch
from model import MGTFNet
from utils.seed import set_seed
from utils import metrics
from data.synthetic_dataset import SyntheticScenes
from utils.preprocessing import normalize_features


def main():
    set_seed()
    root = Path(__file__).resolve().parent
    model = MGTFNet()
    checkpoint = root / "artifacts" / "demo_checkpoint.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        print("Loaded synthetic-demo checkpoint (not a paper checkpoint).")
    else:
        normalization = normalize_features(torch.cat([scene[0] for scene in SyntheticScenes()]))
        model = MGTFNet(normalization=normalization)
        print("No demo checkpoint found: using randomly initialized weights.")
    with np.load(root / "data" / "example_synthetic.npz", allow_pickle=False) as data:
        observed = torch.from_numpy(data["observed_features"])
        future = torch.from_numpy(data["future_positions"])
        ids = torch.from_numpy(data["scene_ids"])
    model.eval()
    with torch.no_grad():
        predicted = model.predict_positions(observed, ids)
    print(f"Observed input shape: {list(observed.shape)}")
    print(f"Number of scenes: {len(torch.unique(ids))}; number of aircraft: {len(observed)}")
    print(f"Predicted trajectory shape: {list(predicted.shape)}")
    for label, function in (("ADE3D", metrics.ade_3d), ("FDE3D", metrics.fde_3d),
                            ("ADE-H", metrics.ade_horizontal), ("FDE-H", metrics.fde_horizontal),
                            ("MAE-V", metrics.mae_vertical), ("FVE", metrics.fve)):
        print(f"{label}: {function(predicted, future):.3f} meters")


if __name__ == "__main__":
    main()
