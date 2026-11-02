"""
AI Waste Classification - CNN (Improved Version)
Classes: cardboard, glass, metal, paper, plastic, trash

Dataset layout:
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/

Run:
python train.py --data_dir dataset --epochs 40
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# -------------------- Configuration --------------------

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -------------------- Reproducibility --------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------- Find and load image paths --------------------

def find_class_folders(data_dir: Path):
    """
    Find the six class folders either directly inside data_dir
    or one directory level below it.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ Dataset: {data_dir.resolve()}")

    possible_parents = [data_dir]
    possible_parents += [p for p in data_dir.iterdir() if p.is_dir()]

    required = set(CLASS_NAMES)

    for parent in possible_parents:
        folders = {
            p.name.lower(): p
            for p in parent.iterdir()
            if p.is_dir()
        }

        if required.issubset(folders.keys()):
            return {name: folders[name] for name in CLASS_NAMES}

    raise FileNotFoundError(
        "ไม่พบโฟลเดอร์ครบทั้ง 6 คลาส\n"
        "ต้องมีชื่อ: cardboard, glass, metal, paper, plastic, trash\n"
        f"ตำแหน่งที่ตรวจสอบ: {data_dir.resolve()}"
    )


def collect_image_paths(data_dir: Path):
    class_folders = find_class_folders(data_dir)

    all_paths = []
    all_labels = []

    print("\n========== DATASET SUMMARY ==========")

    for label, class_name in enumerate(CLASS_NAMES):
        folder = class_folders[class_name]

        image_files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )

        if len(image_files) == 0:
            raise ValueError(f"ไม่พบรูปภาพในโฟลเดอร์ {class_name}")

        all_paths.extend(str(p) for p in image_files)
        all_labels.extend([label] * len(image_files))

        print(f"{class_name:10s}: {len(image_files):4d} images")

    print(f"TOTAL IMAGES: {len(all_paths)}")
    print("=====================================\n")

    return np.array(all_paths), np.array(all_labels, dtype=np.int32)


# -------------------- Image preprocessing --------------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# -------------------- Oversampling (TRAIN SET ONLY) --------------------

def oversample_training_data(paths, labels):
    """
    Balance the training set by randomly oversampling minority classes.
    Validation and test sets are NOT oversampled.
    """
    rng = np.random.default_rng(SEED)
    class_ids, class_counts = np.unique(labels, return_counts=True)

    target_count = int(np.max(class_counts))

    print("Training samples before oversampling:")
    for class_id, count in zip(class_ids, class_counts):
        print(f"  {CLASS_NAMES[int(class_id)]:10s}: {count}")

    balanced_paths = []
    balanced_labels = []

    for class_id in class_ids:
        indices = np.where(labels == class_id)[0]

        sampled_indices = rng.choice(
            indices,
            size=target_count,
            replace=(len(indices) < target_count),
        )

        balanced_paths.extend(paths[sampled_indices])
        balanced_labels.extend(labels[sampled_indices])

    balanced_paths = np.asarray(balanced_paths)
    balanced_labels = np.asarray(balanced_labels, dtype=np.int32)

    order = rng.permutation(len(balanced_paths))
    balanced_paths = balanced_paths[order]
    balanced_labels = balanced_labels[order]

    print("\nTraining samples after oversampling:")
    for class_id, count in zip(*np.unique(balanced_labels, return_counts=True)):
        print(f"  {CLASS_NAMES[int(class_id)]:10s}: {count}")

    print(f"  TOTAL     : {len(balanced_paths)}\n")

    return balanced_paths, balanced_labels


# -------------------- CNN Model --------------------

def build_cnn_model(number_of_classes):
    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="input_image",
    )

    # Data augmentation is applied only during training.
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="improved_waste_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------- Save evaluation outputs --------------------

def save_training_graphs(history, reports_dir: Path):
    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(reports_dir / "accuracy.png", dpi=160)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(reports_dir / "loss.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, reports_dir: Path):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=160)
    plt.close()


# -------------------- Main training process --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train an improved CNN for 6-class waste classification."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Path to the folder containing the six class folders.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
    )

    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports",
    )

    args = parser.parse_args()

    set_seed()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read dataset
    paths, labels = collect_image_paths(data_dir)

    # 2. Split data BEFORE oversampling to prevent data leakage.
    # Train 70%, Validation 15%, Test 15%.
    x_train, x_temp, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    print("========== DATA SPLIT ==========")
    print(f"Train      : {len(x_train)}")
    print(f"Validation : {len(x_val)}")
    print(f"Test       : {len(x_test)}")
    print("================================\n")

    # 3. Oversample TRAIN ONLY
    x_train, y_train = oversample_training_data(x_train, y_train)

    # 4. Create TensorFlow datasets
    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val)
    test_ds = make_dataset(x_test, y_test)

    # 5. Build model
    model = build_cnn_model(len(CLASS_NAMES))
    model.summary()

    # 6. Training callbacks
    model_path = output_dir / "waste_cnn_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 7. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 8. Load best checkpoint and evaluate on untouched test set
    best_model = tf.keras.models.load_model(model_path)

    test_loss, test_accuracy = best_model.evaluate(
        test_ds,
        verbose=1,
    )

    probabilities = best_model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss    : {test_loss:.4f}")
    print("\nClassification Report:")
    print(report)
    print("========================================\n")

    # 9. Save class names and test report
    (output_dir / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_text = (
        f"Test accuracy: {test_accuracy:.4f}\n"
        f"Test loss: {test_loss:.4f}\n\n"
        f"{report}"
    )

    (reports_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    # Save training history for later analysis
    history_json = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    (reports_dir / "training_history.json").write_text(
        json.dumps(history_json, indent=2),
        encoding="utf-8",
    )

    # 10. Save graphs and confusion matrix
    save_training_graphs(history, reports_dir)
    save_confusion_matrix(y_test, predictions, reports_dir)

    print(f"Model saved to   : {model_path.resolve()}")
    print(f"Reports saved to : {reports_dir.resolve()}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
