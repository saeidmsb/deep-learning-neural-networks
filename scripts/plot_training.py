from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = (
    PROJECT_ROOT / "results" / "exercise1_none" / "history.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    history = pd.read_csv(HISTORY_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Training loss
    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        color="tab:blue",
        linewidth=2,
    )
    axes[0].set_title("Training Loss — No Hidden Activation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")

    # Training accuracy: convert fractions to percentages.
    axes[1].plot(
        history["epoch"],
        history["train_accuracy"] * 100,
        color="tab:green",
        linewidth=2,
    )
    axes[1].set_title("Training Accuracy — No Hidden Activation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    for ax in axes:
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = OUTPUT_DIR / "exercise1_none_training.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()