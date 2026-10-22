from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = PROJECT_ROOT / "data" / "mnist_dataset.npz"
OUTPUT_FILE = PROJECT_ROOT / "data" / "mnist_subset.npz"

SEED = 42
SAMPLE_COUNT = 500


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Subset already exists; it was not overwritten: {OUTPUT_FILE}"
        )

    rng = np.random.default_rng(SEED)

    with np.load(SOURCE_FILE, allow_pickle=False) as data:
        train_indices = rng.choice(
            len(data["train_labels"]), SAMPLE_COUNT, replace=False
        )
        test_indices = rng.choice(
            len(data["test_labels"]), SAMPLE_COUNT, replace=False
        )

        train_images = data["train_images"][train_indices]
        train_labels = data["train_labels"][train_indices]
        test_images = data["test_images"][test_indices]
        test_labels = data["test_labels"][test_indices]

    metadata = {
        "source": SOURCE_FILE.name,
        "seed": SEED,
        "sampling": "Uniform random sampling without replacement",
        "train_samples": SAMPLE_COUNT,
        "test_samples": SAMPLE_COUNT,
        "numpy_version": np.__version__,
    }

    np.savez_compressed(
        OUTPUT_FILE,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        metadata=np.array(json.dumps(metadata)),
    )

    print(f"Seed: {SEED}")
    print(f"Training images: {train_images.shape}")
    print(f"Test images: {test_images.shape}")
    print("\nDigit | Train | Test")

    train_counts = np.bincount(train_labels, minlength=10)
    test_counts = np.bincount(test_labels, minlength=10)

    for digit in range(10):
        print(
            f"{digit:5d} | {train_counts[digit]:5d} | "
            f"{test_counts[digit]:4d}"
        )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()