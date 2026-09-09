import json
import os
import sys


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from data_loader import load_data

from preprocessing import (
    preprocess_data,
    standardize_data
)

from split_data import (
    split_dataset
)

from nn_model import (
    train_model,
    predict_model
)

from evaluate import (
    evaluate_model,
    plot_history
)


# -------------------------
# Paths
# -------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# -------------------------
# Experiment Settings
# -------------------------

EPOCHS_LIST = [
    10,
    20,
    30
]

CONFIGS = [
    "small",
    "medium",
    "large"
]

BATCH_SIZE = 16


def main():

    print("=" * 60)

    print(
        "NEURAL NETWORK - "
        "ZOO ANIMALS CLASSIFICATION"
    )

    print("=" * 60)

    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -------------------------
    # Step 1
    # Load Dataset
    # -------------------------

    print(
        "\n[Step 1] Loading dataset..."
    )

    data = load_data(
        DATA_PATH
    )

    print(
        "\nDataset Preview:"
    )

    print(
        data.head()
    )

    # -------------------------
    # Step 2
    # Preprocessing
    # -------------------------

    print(
        "\n[Step 2] Preprocessing..."
    )

    X, y = preprocess_data(
        data
    )

    # Change labels from 1-7 to 0-6
    y = y - 1

    print(
        f"Feature shape: "
        f"{X.shape}"
    )

    print(
        f"Number of samples: "
        f"{len(X)}"
    )

    print(
        f"Number of classes: "
        f"{len(np.unique(y))}"
    )

    # -------------------------
    # Step 3
    # Split Dataset
    # -------------------------

    print(
        "\n[Step 3] Splitting dataset..."
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(
        X,
        y
    )

    # -------------------------
    # Step 4
    # Standardization
    # -------------------------

    print(
        "\n[Step 4] Standardizing data..."
    )

    (
        X_train,
        X_val,
        X_test,
        scaler
    ) = standardize_data(
        X_train,
        X_val,
        X_test
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_val)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # -------------------------
    # Save Dataset
    # -------------------------

    print(
        "\nSaving processed data..."
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "features.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "labels.npy"
        ),
        y
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_val.npy"
        ),
        X_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        ),
        y_test
    )

    # -------------------------
    # Class Names
    # -------------------------

    classes = [
        "Mammal",
        "Bird",
        "Reptile",
        "Fish",
        "Amphibian",
        "Insect",
        "Invertebrate"
    ]

    with open(
        os.path.join(
            OUTPUT_DIR,
            "classes.json"
        ),
        "w"
    ) as file:

        json.dump(
            classes,
            file,
            indent=4
        )

    # -------------------------
    # Step 5
    # Train Models
    # -------------------------

    print(
        "\n[Step 5] Training models..."
    )

    results = []

    best_model = None
    best_history = None
    best_predictions = None
    best_accuracy = 0

    best_config = None
    best_epochs = None

    # Train all configurations
    for config in CONFIGS:

        for epochs in EPOCHS_LIST:

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Configuration: "
                f"{config.upper()}"
            )

            print(
                f"Epochs: "
                f"{epochs}"
            )

            print(
                "=" * 60
            )

            model, history = train_model(
                X_train,
                y_train,
                X_val,
                y_val,
                num_classes=len(classes),
                config=config,
                epochs=epochs,
                batch_size=BATCH_SIZE
            )

            predictions = predict_model(
                model,
                X_test
            )

            accuracy = evaluate_model(
                y_test,
                predictions,
                classes
            )

            # Save experiment result
            results.append(
                {
                    "configuration": config,
                    "epochs": epochs,
                    "accuracy": float(
                        accuracy
                    )
                }
            )

            # Check best model
            if accuracy > best_accuracy:

                best_accuracy = accuracy

                best_model = model

                best_history = history

                best_predictions = predictions

                best_config = config

                best_epochs = epochs

    # -------------------------
    # Step 6
    # Save Best Model
    # -------------------------

    print(
        "\n[Step 6] Saving best model..."
    )

    print(
        f"\nBest Configuration: "
        f"{best_config}"
    )

    print(
        f"Best Epochs: "
        f"{best_epochs}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    # Save model
    best_model.save(
        os.path.join(
            OUTPUT_DIR,
            "nn_model.keras"
        )
    )

    print(
        "\nModel saved successfully."
    )

    # Save best training history
    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in
        best_history.history.items()
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "history.json"
        ),
        "w"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    # -------------------------
    # Step 7
    # Save Evaluation
    # -------------------------

    print(
        "\n[Step 7] Evaluating best model..."
    )

    evaluate_model(
        y_test,
        best_predictions,
        classes,
        save_path=os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    # -------------------------
    # Training History Graph
    # -------------------------

    plot_history(
        best_history,
        os.path.join(
            OUTPUT_DIR,
            "training_history.png"
        )
    )

    # -------------------------
    # Save Experiment Results
    # -------------------------

    with open(
        os.path.join(
            OUTPUT_DIR,
            "results.json"
        ),
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "\nResults saved successfully."
    )

    # -------------------------
    # Final Results
    # -------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EXPERIMENT RESULTS"
    )

    print(
        "=" * 60
    )

    for result in results:

        print(
            f"{result['configuration'].upper():<10} "
            f"| Epochs: "
            f"{result['epochs']:<3} "
            f"| Accuracy: "
            f"{result['accuracy'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()