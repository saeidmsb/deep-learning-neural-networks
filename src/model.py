import torch
from torch import nn


class MNISTNetwork(nn.Module):
    def __init__(self, activation="none"):
        super().__init__()

        self.hidden = nn.Linear(784, 10)
        self.output = nn.Linear(10, 10)

        activations = {
            "none": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation = activations[activation]

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = MNISTNetwork(activation="none")

    # Five artificial images, each represented by 784 pixel values.
    sample_input = torch.zeros(5, 784)

    with torch.no_grad():
        sample_output = model(sample_input)

    print(model)
    print(f"\nInput shape: {tuple(sample_input.shape)}")
    print(f"Output shape: {tuple(sample_output.shape)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    assert sample_output.shape == (5, 10)
    assert torch.isfinite(sample_output).all()

    print("Model checks: PASSED")