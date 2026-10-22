from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

EXPERIMENTS = {
    "none": {
        "title": "No Hidden Activation",
        "runs": {
            "0.1": "exercise1_none",
            "0.01": "exercise1_none_lr001",
        },
    },
    "sigmoid": {
        "title": "Sigmoid",
        "runs": {
            "0.1": "exercise1_sigmoid",
            "0.01": "exercise1_sigmoid_lr001",
        },
    },
    "relu": {
        "title": "ReLU",
        "runs": {
            "0.1": "exercise1_relu",
            "0.01": "exercise1_relu_lr001",
        },
    },
}

STYLES = {
    "0.1": {"color": "tab:blue", "linestyle": "-"},
    "0.01": {"color": "tab:orange", "linestyle": "--"},
}


def load_history(folder_name, activation, learning_rate):
    run_dir = RESULTS_DIR / folder_name

    history = pd.read_csv(run_dir / "history.csv")

    with (run_dir / "summary.json").open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    # Check that the selected files belong to the intended run.
    if summary["activation"] != activation:
        raise ValueError(f"Unexpected activation in {run_dir}")

    if abs(summary["learning_rate"] - float(learning_rate)) > 1e-12:
        raise ValueError(f"Unexpected learning rate in {run_dir}")

    required_columns = {"epoch", "train_loss", "train_accuracy"}
    if not required_columns.issubset(history.columns):
        raise ValueError(f"Missing history columns in {run_dir}")

    history = history.sort_values("epoch").reset_index(drop=True)

    if history["epoch"].tolist() != list(range(101)):
        raise ValueError(
            f"Expected epochs 0 through 100 in {run_dir}"
        )

    if history[list(required_columns)].isna().any().any():
        raise ValueError(f"Missing history values in {run_dir}")

    return history, summary


def main():
    # Read all experiments before creating the figures.
    experiments = {}

    for activation, config in EXPERIMENTS.items():
        experiments[activation] = {}

        for learning_rate, folder_name in config["runs"].items():
            experiments[activation][learning_rate] = load_history(
                folder_name,
                activation,
                learning_rate,
            )

    # Confirm that important comparison settings are identical.
    shared_settings = [
        "architecture",
        "seed",
        "epochs",
        "optimizer",
        "momentum",
        "batch_size",
        "loss_function",
        "metric_timing",
        "accuracy_unit",
        "train_samples",
        "test_samples",
        "torch_version",
    ]

    reference = experiments["none"]["0.1"][1]

    for activation, runs in experiments.items():
        for learning_rate, (_, summary) in runs.items():
            for setting in shared_settings:
                if summary[setting] != reference[setting]:
                    raise ValueError(
                        f"Different {setting}: "
                        f"{activation}, learning rate {learning_rate}"
                    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same loss scale across the three figures.
    max_loss = max(
        history["train_loss"].max()
        for runs in experiments.values()
        for history, _ in runs.values()
    )

    for activation, config in EXPERIMENTS.items():
        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 4.5),
            constrained_layout=True,
        )

        for learning_rate, (history, _) in experiments[activation].items():
            style = STYLES[learning_rate]
            label = f"Learning rate = {learning_rate}"

            axes[0].plot(
                history["epoch"],
                history["train_loss"],
                label=label,
                linewidth=2,
                **style,
            )

            axes[1].plot(
                history["epoch"],
                history["train_accuracy"] * 100,
                label=label,
                linewidth=2,
                **style,
            )

        axes[0].set_title("Training Loss")
        axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].set_ylim(0, max_loss * 1.05)

        axes[1].set_title("Training Accuracy")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100)

        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.suptitle(
            f"{config['title']} — Learning Rate Comparison",
            fontsize=14,
        )

        output_file = (
            OUTPUT_DIR / f"exercise1_{activation}_lr_comparison.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)
        print(f"Saved: {output_file}")

    print("\nAll three comparison figures were saved.")


if __name__ == "__main__":
    main()