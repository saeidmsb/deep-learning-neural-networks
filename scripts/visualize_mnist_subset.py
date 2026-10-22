from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "mnist_subset.npz"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(DATA_FILE, allow_pickle=False) as data:
        train_images = data["train_images"][:10]
        train_labels = data["train_labels"][:10]
        test_images = data["test_images"][:10]
        test_labels = data["test_labels"][:10]

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))

    for row, (images, labels) in enumerate([
        (train_images, train_labels),
        (test_images, test_labels),
    ]):
        for col in range(10):
            axes[row, col].imshow(
                images[col], cmap="gray", vmin=0, vmax=255
            )
            axes[row, col].set_title(f"Label: {labels[col]}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    axes[0, 0].set_ylabel("Train", fontsize=12)
    axes[1, 0].set_ylabel("Test", fontsize=12)

    fig.suptitle("Examples from the selected MNIST subsets")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_file = OUTPUT_DIR / "mnist_train_test_samples.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()