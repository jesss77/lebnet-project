import torch

from src.models.resnet import get_resnet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_resnet18().to(device)

    x = torch.randn(4, 3, 32, 32).to(device)

    with torch.no_grad():
        logits = model(x)

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)


if __name__ == "__main__":
    main()