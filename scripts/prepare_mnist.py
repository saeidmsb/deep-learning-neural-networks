
"""
Prepare the MNIST dataset for neural-network experiments.

Steps:
1. Download MNIST only if the raw files are unavailable.
2. Preserve the official training/test split.
3. Save images, labels, and metadata in a compressed NPZ file.
4. Verify the saved arrays.
"""

from pathlib import Path
import json

import numpy as np
from torchvision import datasets


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "mnist_dataset.npz"

# Create the data directory if it does not exist.
DATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_mnist():
    """Download, save, and verify MNIST."""

    # Reuse the prepared dataset if it already exists and is valid.
    if OUTPUT_FILE.exists():
        try:
            with np.load(OUTPUT_FILE, allow_pickle=False) as saved:
                expected = {
                    "train_images": (60000, 28, 28),
                    "train_labels": (60000,),
                    "test_images": (10000, 28, 28),
                    "test_labels": (10000,),
                }

                if all(
                    key in saved.files
                    and saved[key].shape == shape
                    for key, shape in expected.items()
                ):
                    print(f"Prepared dataset already exists: {OUTPUT_FILE}")
                    print("Existing dataset structure: valid")
                    return

        except (OSError, ValueError, KeyError):
            print("Existing dataset could not be validated; rebuilding it.")

    print("Loading MNIST training data...")
    train = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )

    print("Loading MNIST test data...")
    test = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )

    metadata = {
        "dataset": "MNIST",
        "format": "NumPy compressed archive (NPZ)",
        "image_shape": [28, 28],
        "image_dtype": "uint8",
        "pixel_range": [0, 255],
        "num_classes": 10,
        "train_samples": len(train),
        "test_samples": len(test),
        "source": "torchvision.datasets.MNIST",
        "split_policy": "Official MNIST train/test split preserved",
    }

    print("Saving the prepared dataset...")

    np.savez_compressed(
        OUTPUT_FILE,
        train_images=train.data.numpy(),
        train_labels=train.targets.numpy(),
        test_images=test.data.numpy(),
        test_labels=test.targets.numpy(),
        metadata=np.array(json.dumps(metadata)),
    )

    print("\nVerifying the saved dataset...")

    expected_shapes = {
        "train_images": (60000, 28, 28),
        "train_labels": (60000,),
        "test_images": (10000, 28, 28),
        "test_labels": (10000,),
    }

    with np.load(OUTPUT_FILE, allow_pickle=False) as saved:
        for name, expected_shape in expected_shapes.items():
            array = saved[name]
            assert array.shape == expected_shape, (
                f"{name}: expected {expected_shape}, got {array.shape}"
            )
            print(
                f"{name}: shape={array.shape}, dtype={array.dtype}"
            )

        saved_metadata = json.loads(saved["metadata"].item())
        print("\nMetadata:")
        print(json.dumps(saved_metadata, indent=2))

    size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"File size: {size_mb:.2f} MB")
    print("Verification: PASSED")


if __name__ == "__main__":
    prepare_mnist()
