import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(
    y_test,
    predictions,
    classes,
    save_path=None
):
    """Evaluate Neural Network."""

    labels = list(
        range(
            len(classes)
        )
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\n"
        + "-" * 50
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "-" * 50
    )

    # Classification Report
    print(
        "\nClassification Report:\n"
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)

    # Confusion Matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(
        "\nConfusion Matrix:\n"
    )

    print(matrix)

    if save_path:

        plot_confusion_matrix(
            matrix,
            classes,
            save_path
        )

        print(
            f"\nSaved: "
            f"{save_path}"
        )

    return accuracy


def plot_confusion_matrix(
    matrix,
    classes,
    save_path
):
    """Create Confusion Matrix image."""

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_xticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_yticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    threshold = (
        matrix.max() / 2
        if matrix.max() > 0
        else 0
    )

    for i in range(
        len(classes)
    ):

        for j in range(
            len(classes)
        ):

            color = (
                "white"
                if matrix[i, j] > threshold
                else "black"
            )

            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color=color
            )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def plot_history(
    history,
    save_path
):
    """Plot training and validation accuracy/loss."""

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Accuracy
    axes[0].plot(
        history.history["accuracy"],
        label="Training"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy"
    )

    axes[0].set_title(
        "Training and Validation Accuracy"
    )

    axes[0].legend()

    # Loss
    axes[1].plot(
        history.history["loss"],
        label="Training"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Validation"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].set_title(
        "Training and Validation Loss"
    )

    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)

    print(
        f"Saved: {save_path}"
    )