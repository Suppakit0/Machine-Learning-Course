from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train):

    models = {}

    kernels = {

        "linear": SVC(
            kernel="linear",
            C=1.0
        ),

        "poly": SVC(
            kernel="poly",
            C=1.0,
            degree=3
        ),

        "rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    }

    for name, svm in kernels.items():

        pipeline = Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "svm",
                svm
            )
        ])

        print(
            f"Training {name.upper()} SVM..."
        )

        pipeline.fit(
            X_train,
            y_train
        )

        models[name] = pipeline

    return models


def predict_svm(models, X_test):

    predictions = {}

    for name, model in models.items():

        predictions[name] = model.predict(
            X_test
        )

    return predictions