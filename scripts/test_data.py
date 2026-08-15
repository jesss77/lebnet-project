import torch

from src.data.dataset import get_cifar10


def main():
    train_loader, calibration_loader, test_loader = get_cifar10()

    train_size = len(train_loader.dataset)  # type: ignore[arg-type]
    calibration_size = len(calibration_loader.dataset)  # type: ignore[arg-type]
    test_size = len(test_loader.dataset)  # type: ignore[arg-type]

    print(f"Training samples: {train_size}")
    print(f"Calibration samples: {calibration_size}")
    print(f"Test samples: {test_size}")

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Image device: {images.device}")

    print(f"Image min: {images.min().item():.4f}")
    print(f"Image max: {images.max().item():.4f}")

    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()