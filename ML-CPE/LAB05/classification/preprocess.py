import numpy as np


def preprocess_data(X):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    return X