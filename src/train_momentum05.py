from pathlib import Path
import csv
import json

import torch
from torch import nn

from data_loader import load_mnist_subset
from model import MNISTNetwork


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "exercise2_relu_momentum05"
)

SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.1
ACTIVATION = "relu"
MOMENTUM = 0.5

WEIGHT_EPOCHS = {2, 10, 50, 100}


def evaluate(model, x, y, criterion):
    """Measure loss and accuracy without updating parameters."""
    model.eval()

    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()

    return loss, accuracy


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Preserve any previous results.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}"
        )

    # Same initialization as the previous ReLU experiment.
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    x_train, y_train, x_test, y_test = load_mnist_subset()

    model = MNISTNetwork(activation=ACTIVATION)
    criterion = nn.CrossEntropyLoss()

    # Full-batch gradient descent with ordinary momentum.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    )

    history = []
    selected_weights = []

    # Epoch zero: before any parameter updates.
    initial_loss, initial_accuracy = evaluate(
        model, x_train, y_train, criterion
    )

    history.append({
        "epoch": 0,
        "train_loss": initial_loss,
        "train_accuracy": initial_accuracy,
    })

    print(f"Activation: {ACTIVATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Momentum: {MOMENTUM}")
    print(
        f"Epoch   0 | Loss: {initial_loss:.6f} | "
        f"Accuracy: {initial_accuracy * 100:.2f}%"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_train)
        loss = criterion(logits, y_train)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        loss.backward()
        optimizer.step()

        if not all(
            torch.isfinite(parameter).all().item()
            for parameter in model.parameters()
        ):
            raise RuntimeError(
                f"Non-finite parameters at epoch {epoch}"
            )

        # Record metrics after the parameter update.
        train_loss, train_accuracy = evaluate(
            model, x_train, y_train, criterion
        )

        if not torch.isfinite(torch.tensor(train_loss)).item():
            raise RuntimeError(
                f"Non-finite evaluation loss at epoch {epoch}"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        # Keep recording the same two connections.
        if epoch in WEIGHT_EPOCHS:
            selected_weights.append({
                "epoch": epoch,
                "hidden1_to_digit0": (
                    model.output.weight[0, 0].item()
                ),
                "hidden2_to_digit1": (
                    model.output.weight[1, 1].item()
                ),
            })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch in WEIGHT_EPOCHS
        ):
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_loss:.6f} | "
                f"Accuracy: {train_accuracy * 100:.2f}%"
            )

    # Test evaluation only after all training epochs.
    test_loss, test_accuracy = evaluate(
        model, x_test, y_test, criterion
    )

    summary = {
        "activation": ACTIVATION,
        "architecture": [784, 10, 10],
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": (
            "Full-batch gradient descent with momentum"
        ),
        "momentum": MOMENTUM,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": False,
        "batch_size": len(y_train),
        "loss_function": "CrossEntropyLoss",
        "metric_timing": "After each epoch update",
        "accuracy_unit": "Fraction between 0 and 1",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "torch_version": torch.__version__,
    }

    OUTPUT_DIR.mkdir(parents=True)

    save_csv(OUTPUT_DIR / "history.csv", history)
    save_csv(
        OUTPUT_DIR / "selected_weights.csv",
        selected_weights,
    )

    with (OUTPUT_DIR / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "model.pt",
    )

    print("\nSelected weights after each specified epoch:")

    for row in selected_weights:
        print(
            f"Epoch {row['epoch']:3d} | "
            f"W(hidden1 -> digit0): "
            f"{row['hidden1_to_digit0']:.6f} | "
            f"W(hidden2 -> digit1): "
            f"{row['hidden2_to_digit1']:.6f}"
        )

    print(f"\nTest loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"\nSaved results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()