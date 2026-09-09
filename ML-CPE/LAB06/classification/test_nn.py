import json
import os
import sys

# Windows consoles often default to a legacy codepage (e.g. cp1252) that
# cannot encode non-ASCII characters. Since paths on this machine contain
# Thai characters, force stdout/stderr to UTF-8 so printing them never
# crashes the script.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from tensorflow import keras


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


N_SAMPLES = 4


def test_nn(
    n_samples=N_SAMPLES
):

    # -------------------------
    # Load Model
    # -------------------------

    model = keras.models.load_model(
        os.path.join(
            OUTPUT_DIR,
            "nn_model.keras"
        )
    )

    # -------------------------
    # Load Test Data
    # -------------------------

    X_test = np.load(
        os.path.join(
            OUTPUT_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        )
    )

    # -------------------------
    # Load Class Names
    # -------------------------

    with open(
        os.path.join(
            OUTPUT_DIR,
            "classes.json"
        )
    ) as file:

        classes = json.load(
            file
        )

    # -------------------------
    # Random Samples
    # -------------------------

    indices = np.random.choice(
        len(X_test),
        n_samples,
        replace=False
    )

    X_sample = X_test[
        indices
    ]

    y_sample = y_test[
        indices
    ]

    # -------------------------
    # Prediction
    # -------------------------

    probabilities = model.predict(
        X_sample,
        verbose=0
    )

    predictions = probabilities.argmax(
        axis=1
    )

    confidence = probabilities.max(
        axis=1
    )

    # -------------------------
    # Display Results
    # -------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RANDOM PREDICTION TEST"
    )

    print(
        "=" * 60
    )

    correct_total = 0

    for i in range(
        n_samples
    ):

        predicted_class = classes[
            predictions[i]
        ]

        actual_class = classes[
            y_sample[i]
        ]

        correct = (
            predictions[i]
            == y_sample[i]
        )

        if correct:

            correct_total += 1

        result = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        print(
            f"\nSample "
            f"{i + 1}"
        )

        print(
            f"Predicted: "
            f"{predicted_class}"
        )

        print(
            f"Actual:    "
            f"{actual_class}"
        )

        print(
            f"Confidence: "
            f"{confidence[i] * 100:.2f}%"
        )

        print(
            f"Result: "
            f"{result}"
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Correct Predictions: "
        f"{correct_total}/{n_samples}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    test_nn()