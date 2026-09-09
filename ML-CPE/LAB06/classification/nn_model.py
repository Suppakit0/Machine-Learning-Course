import json
import os

from tensorflow import keras
from tensorflow.keras import layers


# Hidden-layer sizes for each configuration. Input is a flat vector of
# standardized tabular features, so there's no Rescaling/Flatten step here
# (that was only needed for raw image pixels).
ARCHITECTURES = {
    "small": [16, 8],
    "medium": [32, 16, 8],
    "large": [64, 32, 16],
}


def build_model(input_shape, num_classes, config="medium"):
    """Fully-connected neural network (MLP) for tabular features."""

    hidden_units = ARCHITECTURES.get(
        config,
        ARCHITECTURES["medium"]
    )

    model = keras.Sequential(
        [keras.Input(shape=input_shape)]
    )

    for units in hidden_units:

        model.add(
            layers.Dense(units, activation="relu")
        )

        model.add(
            layers.BatchNormalization()
        )

        model.add(
            layers.Dropout(0.3)
        )

    # 1 sigmoid output for 2 classes, softmax otherwise
    model.add(
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        )
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                config="medium", output_dir=None, epochs=30, batch_size=32):
    """Build, train and save the model. Returns (model, history)."""

    model = build_model(X_train.shape[1:], num_classes, config=config)
    model.summary()

    callbacks = [
        # Stop when validation loss stops improving, keep the best weights
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "nn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

        print(f"Saved: {os.path.join(output_dir, 'nn_model.keras')}")

    return model, history


def predict_model(model, X_test):

    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)

    return probabilities.argmax(axis=1)