import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18
from src.training.trainer import train_one_epoch, evaluate


SEED = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDA operations as deterministic as possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")

    train_loader, calibration_loader, test_loader = get_cifar10(
        batch_size=128,
        seed=SEED,
    )

    model = get_resnet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    epochs = 10

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "baseline_resnet18.pt"

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()