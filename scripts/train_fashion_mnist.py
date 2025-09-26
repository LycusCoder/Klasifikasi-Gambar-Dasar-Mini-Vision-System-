"""
Train a simple MLP (Dense) classifier on Fashion-MNIST and export to TFLite.

Usage examples:
  python /app/scripts/train_fashion_mnist.py
  python /app/scripts/train_fashion_mnist.py --epochs 5 --batch-size 64 --output-dir /app/models --model-name fashion_mnist_mlp

This script will:
- Download Fashion-MNIST via tf.keras.datasets
- Normalize pixel values to [0,1]
- Build a simple Sequential model: Flatten -> Dense(512, ReLU) -> Dense(10, Softmax)
- Train and evaluate
- Export model.tflite and Keras .keras file
- Save training metrics to metrics.json (including per-epoch history)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def export_tflite(model: keras.Model, export_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST MLP and export TFLite")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/models",
        help="Directory to save exported model and artifacts",
    )
    parser.add_argument(
        "--model-name", type=str, default="fashion_mnist_mlp", help="Base name for saved model files"
    )

    args = parser.parse_args()

    set_global_seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    model = build_model()
    model.summary()

    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=2,
    )
    train_time_sec = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Save full Keras model
    keras_path = output_dir / f"{args.model_name}.keras"
    model.save(keras_path)

    # Export to TFLite
    tflite_path = output_dir / f"{args.model_name}.tflite"
    export_tflite(model, str(tflite_path))

    # Save metrics (include per-epoch history arrays)
    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_time_sec": round(train_time_sec, 2),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "keras_model_path": str(keras_path),
        "tflite_model_path": str(tflite_path),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }

    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==== Training Complete ====")
    print(json.dumps(metrics, indent=2))
    print("==========================\n")


if __name__ == "__main__":
    main()