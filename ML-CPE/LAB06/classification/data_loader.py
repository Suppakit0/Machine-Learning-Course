import os
import pandas as pd


def load_data(data_path):
    """Load zoo2.csv and zoo3.csv, then combine them."""

    files = [
        "zoo2.csv",
        "zoo3.csv"
    ]

    dataframes = []

    for file_name in files:

        file_path = os.path.join(
            data_path,
            file_name
        )

        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found.")
            continue

        df = pd.read_csv(file_path)

        print(
            f"Loaded {file_name}: "
            f"{len(df)} samples"
        )

        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(
            "No CSV files were loaded."
        )

    # Combine all datasets
    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"\nTotal samples: "
        f"{len(data)}"
    )

    return data