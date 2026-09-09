import numpy as np

from sklearn.preprocessing import StandardScaler


# Feature columns used by the model. "animal_name" is an identifier, not a
# feature, and "class_type" is the label, so both are dropped from X.
FEATURE_COLUMNS = [
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize"
]


def preprocess_data(data):
    """Split the combined zoo dataframe into features (X) and labels (y).

    X is every animal-trait column (all already 0/1, except "legs" which is
    a small integer), and y is "class_type" (1-7).
    """

    X = data[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    y = data["class_type"].to_numpy(
        dtype=np.int64
    )

    return X, y


def standardize_data(X_train, X_val, X_test):
    """Fit a StandardScaler on the training set only, then apply it to all
    three splits so val/test never leak into the fitted mean/std."""

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    return X_train, X_val, X_test, scaler