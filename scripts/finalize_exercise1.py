from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

ACTIVATIONS = {
    "none": {"label": "No activation", "color": "tab:blue"},
    "sigmoid": {"label": "Sigmoid", "color": "tab:orange"},
    "relu": {"label": "ReLU", "color": "tab:green"},
}

RUNS = {
    "0.1": {
        "none": "exercise1_none",
        "sigmoid": "exercise1_sigmoid",
        "relu": "exercise1_relu",
    },
    "0.01": {
        "none": "exercise1_none_lr001",
        "sigmoid": "exercise1_sigmoid_lr001",
        "relu": "exercise1_relu_lr001",
    },
}

WEIGHT_EPOCHS = [2, 10, 50, 100]


def load_run(folder_name, activation, learning_rate):
    run_dir = RESULTS_DIR / folder_name

    history = pd.read_csv(run_dir / "history.csv")
    weights = pd.read_csv(run_dir / "selected_weights.csv")

    with (run_dir / "summary.json").open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    if summary["activation"] != activation:
        raise ValueError(f"Unexpected activation: {run_dir}")

    if abs(summary["learning_rate"] - float(learning_rate)) > 1e-12:
        raise ValueError(f"Unexpected learning rate: {run_dir}")

    history = history.sort_values("epoch").reset_index(drop=True)
    weights = weights.sort_values("epoch").reset_index(drop=True)

    if history["epoch"].tolist() != list(range(101)):
        raise ValueError(f"Expected epochs 0 through 100: {run_dir}")

    if weights["epoch"].tolist() != WEIGHT_EPOCHS:
        raise ValueError(f"Unexpected weight recording epochs: {run_dir}")

    return history, weights, summary


def main():
    experiments = {}
    summary_rows = []
    weight_tables = []

    # Load all six runs before writing outputs.
    for learning_rate, runs in RUNS.items():
        experiments[learning_rate] = {}

        for activation, folder_name in runs.items():
            history, weights, summary = load_run(
                folder_name, activation, learning_rate
            )

            experiments[learning_rate][activation] = (
                history, weights, summary
            )

            summary_rows.append({
                "activation": activation,
                "learning_rate": float(learning_rate),
                "train_loss": summary["final_train_loss"],
                "train_accuracy_percent": (
                    summary["final_train_accuracy"] * 100
                ),
                "test_loss": summary["test_loss"],
                "test_accuracy_percent": (
                    summary["test_accuracy"] * 100
                ),
            })

            weights = weights.copy()
            weights.insert(0, "activation", activation)
            weights.insert(0, "learning_rate", float(learning_rate))
            weight_tables.append(weights)

    # Verify common experimental settings.
    reference = experiments["0.1"]["none"][2]

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

    for runs in experiments.values():
        for _, _, summary in runs.values():
            for setting in shared_settings:
                if summary[setting] != reference[setting]:
                    raise ValueError(
                        f"Inconsistent comparison setting: {setting}"
                    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Identical axis scales make the two figures comparable.
    max_loss = max(
        history["train_loss"].max()
        for runs in experiments.values()
        for history, _, _ in runs.values()
    )

    for learning_rate, runs in experiments.items():
        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 4.5),
            constrained_layout=True,
        )

        for activation, (history, _, _) in runs.items():
            style = ACTIVATIONS[activation]

            axes[0].plot(
                history["epoch"],
                history["train_loss"],
                label=style["label"],
                color=style["color"],
                linewidth=2,
            )

            axes[1].plot(
                history["epoch"],
                history["train_accuracy"] * 100,
                label=style["label"],
                color=style["color"],
                linewidth=2,
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
            f"Activation Comparison — Learning Rate = {learning_rate}",
            fontsize=14,
        )

        tag = learning_rate.replace(".", "")
        output_file = (
            FIGURES_DIR / f"exercise1_activations_lr{tag}.png"
        )

        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_file}")

    summary_table = pd.DataFrame(summary_rows)
    weights_table = pd.concat(weight_tables, ignore_index=True)

    summary_path = TABLES_DIR / "exercise1_summary.csv"
    weights_path = TABLES_DIR / "exercise1_weights_all.csv"

    summary_table.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    weights_table.to_csv(
        weights_path, index=False, encoding="utf-8-sig"
    )

    print("\nFinal results:")
    print(summary_table.to_string(index=False, float_format="%.6f"))

    # One compact comparison table per learning rate.
    for learning_rate in RUNS:
        subset = weights_table[
            weights_table["learning_rate"] == float(learning_rate)
        ]

        comparison = subset.pivot(
            index="epoch",
            columns="activation",
            values=["hidden1_to_digit0", "hidden2_to_digit1"],
        )

        comparison = comparison.reindex(
            columns=pd.MultiIndex.from_product([
                ["hidden1_to_digit0", "hidden2_to_digit1"],
                list(ACTIVATIONS),
            ])
        )

        comparison.columns = [
            f"{weight}_{activation}"
            for weight, activation in comparison.columns
        ]

        tag = learning_rate.replace(".", "")
        path = TABLES_DIR / f"exercise1_weights_lr{tag}.csv"

        comparison.to_csv(path, encoding="utf-8-sig")

        print(f"\nSelected weights — learning rate {learning_rate}:")
        print(comparison.to_string(float_format="%.6f"))
        print(f"Saved: {path}")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {weights_path}")
    print("\nExercise 1 figures and tables are ready.")


if __name__ == "__main__":
    main()