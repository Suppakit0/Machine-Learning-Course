import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, kernel,
                   save_path=None):

    accuracy = accuracy_score(y_test, predictions)

    print("\n----------------------------------------")
    print(f"{kernel.upper()} SVM")
    print("----------------------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=classes,
            target_names=[str(c) for c in classes],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    print("Confusion Matrix:")
    print(matrix)

    if save_path:
        plot_confusion_matrix(
            matrix,
            classes,
            kernel,
            save_path
        )

    return accuracy


def plot_confusion_matrix(
    matrix,
    classes,
    kernel,
    save_path
):

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(matrix, cmap="Blues")

    labels = [str(c) for c in classes]

    ax.set_xticks(
        np.arange(len(classes)),
        labels
    )

    ax.set_yticks(
        np.arange(len(classes)),
        labels
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_title(
        f"Confusion Matrix - {kernel.upper()} SVM"
    )

    threshold = matrix.max() / 2

    for i in range(len(classes)):
        for j in range(len(classes)):

            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > threshold
                    else "black"
                )
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)