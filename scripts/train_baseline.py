import torch
import torch.nn as nn
from torch.optim import SGD

from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18
from src.training.trainer import train_one_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader, calibration_loader, test_loader = get_cifar10(
        batch_size=128
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

    torch.save(
        model.state_dict(),
        "results/baseline_resnet18.pt",
    )

    print("Model saved to results/baseline_resnet18.pt")


if __name__ == "__main__":
    main()