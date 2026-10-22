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
    {
        "label": "GD",
        "folder": "exercise1_relu",
        "color": "tab:blue",
        "linestyle": "-",
    },
    {
        "label": "Adam",
        "folder": "exercise2_relu_adam",
        "color": "tab:orange",
        "linestyle": "--",
    },
]

SHARED_SETTINGS = [
    "activation",
    "architecture",
    "seed",
    "epochs",
    "learning_rate",
    "batch_size",
    "loss_function",
    "metric_timing",
    "accuracy_unit",
    "train_samples",
    "test_samples",
    "torch_version",
]


def load_run(config):
    run_dir = RESULTS_DIR / config["folder"]

    history = pd.read_csv(run_dir / "history.csv")

    with (run_dir / "summary.json").open(
        encoding="utf-8"
    ) as file:
        summary = json.load(file)

    if summary["activation"] != "relu":
        raise ValueError(f"Expected ReLU: {run_dir}")

    if not np.isclose(summary["learning_rate"], 0.1):
        raise ValueError(
            f"Expected learning rate 0.1: {run_dir}"
        )

    if config["label"] == "GD":
        if summary["optimizer"] != "Full-batch gradient descent":
            raise ValueError(f"Unexpected GD optimizer: {run_dir}")

        if not np.isclose(summary["momentum"], 0.0):
            raise ValueError(f"GD must have zero momentum: {run_dir}")

    elif summary["optimizer"] != "Adam":
        raise ValueError(f"Expected Adam: {run_dir}")

    columns = ["epoch", "train_loss", "train_accuracy"]

    if not set(columns).issubset(history.columns):
        raise ValueError(f"Missing history columns: {run_dir}")

    history = history.sort_values("epoch").reset_index(drop=True)

    if history["epoch"].tolist() != list(range(101)):
        raise ValueError(
            f"Expected epochs 0 through 100: {run_dir}"
        )

    if not np.isfinite(history[columns].to_numpy()).all():
        raise ValueError(f"Non-finite values: {run_dir}")

    if (history["train_loss"] <= 0).any():
        raise ValueError(
            f"Expected positive losses for log plot: {run_dir}"
        )

    if not history["train_accuracy"].between(0, 1).all():
        raise ValueError(f"Invalid accuracy values: {run_dir}")

    for history_key, summary_key in [
        ("train_loss", "final_train_loss"),
        ("train_accuracy", "final_train_accuracy"),
    ]:
        if not np.isclose(
            history.iloc[-1][history_key],
            summary[summary_key],
        ):
            raise ValueError(
                f"History and summary mismatch: {run_dir}"
            )

    return history, summary


def first_epoch_at_accuracy(history, threshold):
    reached = history.loc[
        history["train_accuracy"] >= threshold,
        "epoch",
    ]

    return None if reached.empty else int(reached.iloc[0])


def main():
    experiments = []

    for config in RUNS:
        history, summary = load_run(config)
        experiments.append({
            **config,
            "history": history,
            "summary": summary,
        })

    reference = experiments[0]["summary"]

    for experiment in experiments[1:]:
        for setting in SHARED_SETTINGS:
            if experiment["summary"][setting] != reference[setting]:
                raise ValueError(
                    f"Inconsistent comparison setting: {setting}"
                )

    # Same model and initialization should give the same epoch-zero metrics.
    initial_reference = experiments[0]["history"].iloc[0]

    for experiment in experiments[1:]:
        initial = experiment["history"].iloc[0]

        for metric in ["train_loss", "train_accuracy"]:
            if not np.isclose(
                initial[metric],
                initial_reference[metric],
            ):
                raise ValueError(
                    f"Different epoch-zero {metric}"
                )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Main training comparison.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    for experiment in experiments:
        history = experiment["history"]

        style = {
            "label": experiment["label"],
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
        "GD vs Adam — ReLU, Learning Rate = 0.1"
    )

    training_path = (
        FIGURES_DIR / "exercise2_gd_adam_training.png"
    )
    fig.savefig(training_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Additional logarithmic loss plot.
    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        constrained_layout=True,
    )

    for experiment in experiments:
        history = experiment["history"]

        ax.semilogy(
            history["epoch"],
            history["train_loss"],
            label=experiment["label"],
            color=experiment["color"],
            linestyle=experiment["linestyle"],
            linewidth=2,
        )

    ax.set_title("GD vs Adam — Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss (log scale)")
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    log_path = (
        FIGURES_DIR / "exercise2_gd_adam_loss_log.png"
    )
    fig.savefig(log_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Final metrics and learning progress.
    rows = []

    for experiment in experiments:
        history = experiment["history"]
        summary = experiment["summary"]

        rows.append({
            "optimizer": experiment["label"],
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
            "first_epoch_train_accuracy_90_percent": (
                first_epoch_at_accuracy(history, 0.90)
            ),
            "first_epoch_train_accuracy_95_percent": (
                first_epoch_at_accuracy(history, 0.95)
            ),
            "training_loss_increase_steps": int(
                (history["train_loss"].diff().dropna() > 0).sum()
            ),
        })

    results = pd.DataFrame(rows)

    for column in [
        "first_epoch_train_accuracy_90_percent",
        "first_epoch_train_accuracy_95_percent",
    ]:
        results[column] = pd.array(
            results[column],
            dtype="Int64",
        )

    summary_path = (
        TABLES_DIR / "exercise2_gd_adam_summary.csv"
    )

    results.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        na_rep="Not reached",
    )

    # Combined history for the report and later analysis.
    combined = pd.DataFrame({
        "epoch": experiments[0]["history"]["epoch"],
    })

    for experiment in experiments:
        tag = experiment["label"].lower()
        history = experiment["history"]

        combined[f"train_loss_{tag}"] = (
            history["train_loss"].to_numpy()
        )
        combined[f"train_accuracy_percent_{tag}"] = (
            history["train_accuracy"].to_numpy() * 100
        )

    history_path = (
        TABLES_DIR / "exercise2_gd_adam_history.csv"
    )

    combined.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinal GD vs Adam comparison:")
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