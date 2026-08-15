import json
import os

import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
N_SAMPLES = 10


def test_svm(n_samples=N_SAMPLES):

    # Load models
    models = {
        "linear": joblib.load(
            f"{OUTPUT_DIR}/svm_linear.pkl"
        ),

        "poly": joblib.load(
            f"{OUTPUT_DIR}/svm_poly.pkl"
        ),

        "rbf": joblib.load(
            f"{OUTPUT_DIR}/svm_rbf.pkl"
        )
    }

    # Load test data
    X_test = np.load(
        f"{OUTPUT_DIR}/X_test.npy"
    )

    y_test = np.load(
        f"{OUTPUT_DIR}/y_test.npy"
    )

    with open(
        f"{OUTPUT_DIR}/classes.json"
    ) as f:

        classes = json.load(f)

    n_samples = min(
        n_samples,
        len(X_test)
    )

    # Random samples
    index = np.random.choice(
        len(X_test),
        n_samples,
        replace=False
    )

    X_sample = X_test[index]
    y_sample = y_test[index]

    print("\n========================================")
    print("SVM PREDICTION TEST")
    print("========================================")

    for i in range(n_samples):

        print(f"\nSample {i + 1}")

        print(
            f"True Class: "
            f"{y_sample[i]}"
        )

        for kernel, model in models.items():

            prediction = model.predict(
                X_sample[i].reshape(1, -1)
            )[0]

            result = (
                "OK"
                if prediction == y_sample[i]
                else "WRONG"
            )

            print(
                f"{kernel.upper():<10} "
                f"Prediction: {prediction:<3} "
                f"{result}"
            )


if __name__ == "__main__":
    test_svm()