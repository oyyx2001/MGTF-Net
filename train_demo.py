"""Small CPU-only execution demo; not a reproduction of paper accuracy."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import config
from model import MGTFNet, multi_horizon_loss
from data.synthetic_dataset import SyntheticScenes, collate_scenes
from utils.preprocessing import normalize_features, to_core_inputs
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    set_seed(config.SEED)
    train = SyntheticScenes(12, config.SEED)
    validation = SyntheticScenes(4, config.SEED + 1)
    normalization = normalize_features(torch.cat([scene[0] for scene in train]))
    model = MGTFNet(normalization=normalization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-5)
    loaders = [DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=(i == 0), collate_fn=collate_scenes)
               for i, dataset in enumerate((train, validation))]
    checkpoint = Path(__file__).resolve().parent / "artifacts" / "demo_checkpoint.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        values = []
        for training, loader in zip((True, False), loaders):
            model.train(training)
            total, count = 0.0, 0
            with torch.set_grad_enabled(training):
                for observed, future, ids in loader:
                    rel, pos, boundaries = to_core_inputs(observed, ids)
                    loss = multi_horizon_loss(model(rel, pos, boundaries), pos, future.transpose(0, 1) / 1000.0, config.HORIZONS)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP, error_if_nonfinite=True)
                        optimizer.step()
                    total += loss.item() * len(observed)
                    count += len(observed)
            values.append(total / count)
        scheduler.step(values[1])
        if values[1] < best:
            best = values[1]
            torch.save({"state_dict": model.state_dict(), "seed": config.SEED, "epoch": epoch + 1}, checkpoint)
        print(f"Epoch {epoch + 1}/{args.epochs}: train loss={values[0]:.6f}, validation loss={values[1]:.6f} (native km objective)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Saved best synthetic-demo checkpoint: artifacts/demo_checkpoint.pt")


if __name__ == "__main__":
    main()
