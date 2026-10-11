import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.ticker import MaxNLocator

def evaluate_model(y_test, predictions, classes, save_path=None):
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    report = classification_report(
        y_test, predictions, labels=labels, target_names=classes, zero_division=0
    )
    print(report)
    print("Confusion Matrix:")
    
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy

def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    
    # เพิ่มแถบสเกลบอกระดับสี (Colorbar)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Number of Predictions", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history.history["accuracy"]) + 1)

    axes[0].plot(epochs, history.history["accuracy"], label="train", marker='o', markersize=4)
    axes[0].plot(epochs, history.history["val_accuracy"], label="validation", marker='o', markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    axes[1].plot(epochs, history.history["loss"], label="train", marker='o', markersize=4)
    axes[1].plot(epochs, history.history["val_loss"], label="validation", marker='o', markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")