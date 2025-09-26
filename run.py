#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR_DEFAULT = REPO_ROOT / "models"


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_setup(_: argparse.Namespace) -> None:
    script = SCRIPTS_DIR / "setup_ml_env.sh"
    if not script.exists():
        print("Setup script tidak ditemukan:", script)
        sys.exit(1)
    exit_code = _run(["bash", str(script)])
    sys.exit(exit_code)


def ensure_tf_available() -> bool:
    try:
        import importlib
        return importlib.util.find_spec("tensorflow") is not None
    except Exception:
        return False


def cmd_train(args: argparse.Namespace) -> None:
    if not ensure_tf_available():
        print("TensorFlow belum terinstal. Jalankan: bash scripts/setup_ml_env.sh")
        sys.exit(1)
    train_script = SCRIPTS_DIR / "train_fashion_mnist.py"
    if not train_script.exists():
        print("Script training tidak ditemukan:", train_script)
        sys.exit(1)
    output_dir = Path(args.output_dir or MODELS_DIR_DEFAULT)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_script),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
        "--model-name",
        args.model_name,
    ]
    code = _run(cmd)
    if code != 0:
        sys.exit(code)

    # Tampilkan ringkasan metrics bila ada
    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text())
        print("\n==== Ringkasan Training ====")
        print(json.dumps(data, indent=2))
        print("===========================\n")
    else:
        print("Metrics tidak ditemukan di:", metrics_path)


def cmd_verify(args: argparse.Namespace) -> None:
    if not ensure_tf_available():
        print("TensorFlow belum terinstal. Jalankan: bash scripts/setup_ml_env.sh")
        sys.exit(1)
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    model_path = Path(args.model)
    if not model_path.exists():
        print("Model TFLite tidak ditemukan:", model_path)
        sys.exit(1)

    # Load dataset sample untuk verifikasi yang lebih nyata
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = x_test[:1]
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    pred = int(out.argmax())
    true_label = int(y_test[0])
    print({
        "predicted_class": pred,
        "true_label": true_label,
        "probs_sum": float(out.sum()),
        "output_shape": tuple(out.shape),
    })


def cmd_paths(args: argparse.Namespace) -> None:
    models_dir = Path(args.dir or MODELS_DIR_DEFAULT)
    if not models_dir.exists():
        print("Folder model tidak ditemukan:", models_dir)
        sys.exit(1)
    metrics = sorted(models_dir.glob("*_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metrics:
        print("Belum ada metrics di:", models_dir)
        sys.exit(0)
    latest = metrics[0]
    data = json.loads(latest.read_text())
    print("Metrics terakhir:")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Runner serbaguna untuk Mini-Vision System (Lycus)")
    sub = parser.add_subparsers(required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Setup environment ML (install TF dan update requirements)")
    p_setup.set_defaults(func=cmd_setup)

    # train
    p_train = sub.add_parser("train", help="Train Fashion-MNIST dan ekspor TFLite")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--output-dir", type=str, default=str(MODELS_DIR_DEFAULT))
    p_train.add_argument("--model-name", type=str, default="fashion_mnist_mlp")
    p_train.set_defaults(func=cmd_train)

    # verify
    p_verify = sub.add_parser("verify", help="Inference cepat pakai TFLite pada sampel dataset test")
    p_verify.add_argument("--model", type=str, required=True, help="Path ke file .tflite")
    p_verify.set_defaults(func=cmd_verify)

    # paths
    p_paths = sub.add_parser("paths", help="Tampilkan path artefak berdasarkan metrics terbaru")
    p_paths.add_argument("--dir", type=str, default=str(MODELS_DIR_DEFAULT))
    p_paths.set_defaults(func=cmd_paths)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()