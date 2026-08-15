import json
import os

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from data_load import load_data
from preprocess import preprocess_data
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")  

TEST_SIZE = 0.2


def main():

    print("=" * 60)
    print("SVM Classification: Zoo Animals")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================
    # Step 1: Load Dataset
    # =========================================

    print("\n[Step 1] Loading dataset...")

    X, y, classes = load_data(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Total samples : {len(X)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Classes       : {classes}")

    np.save(
        f"{OUTPUT_DIR}/features.npy",
        X
    )

    np.save(
        f"{OUTPUT_DIR}/labels.npy",
        y
    )

    with open(
        f"{OUTPUT_DIR}/classes.json",
        "w"
    ) as f:
        json.dump(
            [int(c) for c in classes],
            f
        )

    # =========================================
    # Step 2: Preprocessing
    # =========================================

    print("\n[Step 2] Preprocessing data...")

    X = preprocess_data(X)

    print(f"Feature shape: {X.shape}")

    # =========================================
    # Step 3: Split Dataset
    # =========================================

    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        TEST_SIZE
    )

    np.save(
        f"{OUTPUT_DIR}/X_train.npy",
        X_train
    )

    np.save(
        f"{OUTPUT_DIR}/X_test.npy",
        X_test
    )

    np.save(
        f"{OUTPUT_DIR}/y_train.npy",
        y_train
    )

    np.save(
        f"{OUTPUT_DIR}/y_test.npy",
        y_test
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # =========================================
    # Step 4: Train SVM
    # =========================================

    print("\n[Step 4] Training SVM models...")

    models = train_svm(
        X_train,
        y_train
    )

    for name, model in models.items():

        joblib.dump(
            model,
            f"{OUTPUT_DIR}/svm_{name}.pkl"
        )

    print("\nSVM training completed.")

    # =========================================
    # Step 5: Prediction
    # =========================================

    print("\n[Step 5] Predicting test data...")

    predictions = predict_svm(
        models,
        X_test
    )

    # =========================================
    # Step 6: Evaluation
    # =========================================

    print("\n[Step 6] Evaluating models...")

    results = {}

    for name in models:

        accuracy = evaluate_model(
            y_test,
            predictions[name],
            classes,
            name,
            save_path=(
                f"{OUTPUT_DIR}/"
                f"confusion_matrix_{name}.png"
            )
        )

        results[name] = accuracy

    # =========================================
    # Step 7: Compare Accuracy
    # =========================================

    print("\n" + "=" * 60)
    print("SVM KERNEL COMPARISON")
    print("=" * 60)

    for kernel, accuracy in results.items():

        print(
            f"{kernel.upper():<12} "
            f"{accuracy * 100:.2f}%"
        )

    # =========================================
    # Accuracy Graph
    # =========================================

    kernels = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))

    plt.bar(
        kernels,
        accuracies
    )

    plt.title(
        "SVM Kernel Accuracy Comparison"
    )

    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1.0)

    for i, value in enumerate(accuracies):

        plt.text(
            i,
            value + 0.02,
            f"{value * 100:.2f}%",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/accuracy_comparison.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved: outputs/accuracy_comparison.png"
    )


if __name__ == "__main__":
    main()