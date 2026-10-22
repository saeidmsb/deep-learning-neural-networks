from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "mnist_subset.npz"


def load_mnist_subset():
    """Load the selected samples and prepare them for the network."""
    with np.load(DATA_FILE, allow_pickle=False) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"]

    # Convert each image to 784 floating-point values in [0, 1].
    x_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    # Convert NumPy arrays to PyTorch tensors.
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(train_labels.astype(np.int64))
    y_test = torch.from_numpy(test_labels.astype(np.int64))

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist_subset()

    for name, tensor in [
        ("x_train", x_train),
        ("y_train", y_train),
        ("x_test", x_test),
        ("y_test", y_test),
    ]:
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    print(f"Train pixel range: {x_train.min().item()} to {x_train.max().item()}")
    print(f"Test pixel range: {x_test.min().item()} to {x_test.max().item()}")

    assert x_train.shape == (500, 784)
    assert x_test.shape == (500, 784)
    assert y_train.shape == (500,)
    assert y_test.shape == (500,)
    assert torch.isfinite(x_train).all()
    assert torch.isfinite(x_test).all()
    assert ((x_train >= 0) & (x_train <= 1)).all()
    assert ((x_test >= 0) & (x_test <= 1)).all()
    assert ((y_train >= 0) & (y_train <= 9)).all()
    assert ((y_test >= 0) & (y_test <= 9)).all()

    print("Preprocessing checks: PASSED")