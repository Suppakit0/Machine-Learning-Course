import os
import pandas as pd
import numpy as np


def load_data(data_path):

    files = [
        os.path.join(data_path, "zoo2.csv"),
        os.path.join(data_path, "zoo3.csv")
    ]

    dataframes = []

    for file in files:

        if not os.path.exists(file):
            raise FileNotFoundError(
                f"File not found: {file}"
            )

        df = pd.read_csv(file)

        dataframes.append(df)

        print(
            f"Loaded: {file} "
            f"({len(df)} samples)"
        )

    # Combine datasets
    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    # Remove duplicates
    data = data.drop_duplicates()

    # Remove missing values
    data = data.dropna()

    print("\nDataset combined successfully.")
    print(f"Total samples: {len(data)}")

    # Remove name and target
    X = data.drop(
        columns=["animal_name", "class_type"]
    )

    # Target
    y = data["class_type"]

    # Convert X
    X = X.to_numpy(
        dtype=np.float32
    )

    # Convert y
    y = y.to_numpy()

    classes = sorted(
        np.unique(y).tolist()
    )

    return X, y, classes