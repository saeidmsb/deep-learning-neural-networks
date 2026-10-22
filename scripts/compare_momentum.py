from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

RUNS = [
    (0.5, "exercise2_relu_momentum05", "tab:blue", "-"),
    (0.9, "exercise2_relu_momentum09", "tab:orange", "--"),
    (1.0, "exercise2_relu_momentum1", "tab:green", "-."),
]

SHARED_SETTINGS = [
    "activation",
    "architecture",
    "seed",
    "epochs",
    "learning_rate",
    "optimizer",
    "dampening",
    "weight_decay",
    "nesterov",
    "batch_size",
    "loss_function",
    "metric_timing",
    "accuracy_unit",
    "train_samples",
    "test_samples",
    "torch_version",
]


def load_run(momentum, folder_name):
    run_dir = RESULTS_DIR / folder_name

    history = pd.read_csv(run_dir / "history.csv")

    with (run_dir / "summary.json").open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    if not np.isclose(summary["momentum"], momentum):
        raise ValueError(f"Unexpected momentum: {run_dir}")

    if summary["activation"] != "relu":
        raise ValueError(f"Expected ReLU: {run_dir}")

    if not np.isclose(summary["learning_rate"], 0.1):
        raise ValueError(f"Expected learning rate 0.1: {run_dir}")

    columns = ["epoch", "train_loss", "train_accuracy"]

    if not set(columns).issubset(history.columns):
        raise ValueError(f"Missing history columns: {run_dir}")

    history = history.sort_values("epoch").reset_index(drop=True)

    if history["epoch"].tolist() != list(range(101)):
        raise ValueError(f"Expected epochs 0 through 100: {run_dir}")

    if not np.isfinite(history[columns].to_numpy()).all():
        raise ValueError(f"Non-finite history values: {run_dir}")

    if not history["train_accuracy"].between(0, 1).all():
        raise ValueError(f"Invalid accuracy values: {run_dir}")

    if (history["train_loss"] < 0).any():
        raise ValueError(f"Invalid loss values: {run_dir}")

    if not np.isclose(
        history.iloc[-1]["train_loss"],
        summary["final_train_loss"],
    ):
        raise ValueError(f"Final loss mismatch: {run_dir}")

    if not np.isclose(
        history.iloc[-1]["train_accuracy"],
        summary["final_train_accuracy"],
    ):
        raise ValueError(f"Final accuracy mismatch: {run_dir}")

    return history, summary


def first_epoch_at_accuracy(history, threshold):
    reached = history.loc[
        history["train_accuracy"] >= threshold,
        "epoch",
    ]

    if reached.empty:
        return None

    return int(reached.iloc[0])


def main():
    experiments = []

    # Read and validate all three runs.
    for momentum, folder_name, color, linestyle in RUNS:
        history, summary = load_run(momentum, folder_name)

        experiments.append({
            "momentum": momentum,
            "history": history,
            "summary": summary,
            "color": color,
            "linestyle": linestyle,
        })

    reference = experiments[0]["summary"]

    for experiment in experiments[1:]:
        for setting in SHARED_SETTINGS:
            if experiment["summary"][setting] != reference[setting]:
                raise ValueError(
                    f"Inconsistent setting: {setting}"
                )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: training loss and accuracy.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    for experiment in experiments:
        history = experiment["history"]

        style = {
            "label": f"Momentum = {experiment['momentum']:.1f}",
            "color": experiment["color"],
            "linestyle": experiment["linestyle"],
            "linewidth": 2,
        }

        axes[0].plot(
            history["epoch"],
            history["train_loss"],
            **style,
        )

        axes[1].plot(
            history["epoch"],
            history["train_accuracy"] * 100,
            **style,
        )

    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_ylim(bottom=0)

    axes[1].set_title("Training Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 102)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Momentum Comparison — ReLU, Learning Rate = 0.1"
    )

    training_path = (
        FIGURES_DIR / "exercise2_momentum_training.png"
    )
    fig.savefig(training_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: logarithmic loss scale reveals small losses
    # and fluctuations near the end of training.
    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        constrained_layout=True,
    )

    for experiment in experiments:
        history = experiment["history"]

        if (history["train_loss"] <= 0).any():
            raise ValueError(
                "Logarithmic loss plot requires positive losses."
            )

        ax.semilogy(
            history["epoch"],
            history["train_loss"],
            label=f"Momentum = {experiment['momentum']:.1f}",
            color=experiment["color"],
            linestyle=experiment["linestyle"],
            linewidth=2,
        )

    ax.set_title(
        "Training Loss — Logarithmic Scale"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss (log scale)")
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    log_path = (
        FIGURES_DIR / "exercise2_momentum_loss_log.png"
    )
    fig.savefig(log_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Build the final results table.
    rows = []

    for experiment in experiments:
        history = experiment["history"]
        summary = experiment["summary"]

        loss_increases = int(
            (history["train_loss"].diff().dropna() > 0).sum()
        )

        rows.append({
            "momentum": experiment["momentum"],
            "learning_rate": summary["learning_rate"],
            "train_loss": summary["final_train_loss"],
            "train_accuracy_percent": (
                summary["final_train_accuracy"] * 100
            ),
            "test_loss": summary["test_loss"],
            "test_accuracy_percent": (
                summary["test_accuracy"] * 100
            ),
            "accuracy_gap_percentage_points": (
                summary["final_train_accuracy"]
                - summary["test_accuracy"]
            ) * 100,
            "first_epoch_train_accuracy_95_percent": (
                first_epoch_at_accuracy(history, 0.95)
            ),
            "first_epoch_train_accuracy_100_percent": (
                first_epoch_at_accuracy(history, 1.0)
            ),
            "training_loss_increase_steps": loss_increases,
        })

    results = pd.DataFrame(rows)

    epoch_columns = [
        "first_epoch_train_accuracy_95_percent",
        "first_epoch_train_accuracy_100_percent",
    ]

    for column in epoch_columns:
        results[column] = pd.array(
            results[column],
            dtype="Int64",
        )

    summary_path = (
        TABLES_DIR / "exercise2_momentum_summary.csv"
    )

    results.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        na_rep="Not reached",
    )

    # Save all training curves together for later use.
    combined_history = pd.DataFrame({
        "epoch": experiments[0]["history"]["epoch"],
    })

    for experiment in experiments:
        tag = f"{experiment['momentum']:.1f}".replace(".", "")
        history = experiment["history"]

        combined_history[f"train_loss_m{tag}"] = (
            history["train_loss"].to_numpy()
        )
        combined_history[f"train_accuracy_percent_m{tag}"] = (
            history["train_accuracy"].to_numpy() * 100
        )

    history_path = (
        TABLES_DIR / "exercise2_momentum_history.csv"
    )

    combined_history.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinal momentum comparison:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
            na_rep="Not reached",
        )
    )

    print("\nSaved files:")
    for path in [
        training_path,
        log_path,
        summary_path,
        history_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()