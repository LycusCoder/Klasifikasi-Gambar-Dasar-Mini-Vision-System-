#!/usr/bin/env bash
set -euo pipefail

echo "[Setup] Instalasi TensorFlow CPU (2.17.0) dan sinkronisasi requirements..."

# Instal TensorFlow (CPU)
python - <<'PY'
import sys, subprocess
pkgs = [
    ("tensorflow", "2.17.0"),
]
for name, ver in pkgs:
    print(f"Installing {name}=={ver} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", f"{name}=={ver}"])
PY

# Update requirements.txt sesuai lingkungan saat ini
echo "[Setup] Memperbarui backend/requirements.txt (pip freeze) ..."
pip freeze > /app/backend/requirements.txt

# Restart backend via supervisor (jika tersedia)
if command -v sudo >/dev/null 2>&1 && command -v supervisorctl >/dev/null 2>&1; then
  echo "[Setup] Restart backend via supervisor ..."
  sudo supervisorctl restart backend || true
else
  echo "[Setup] Lewati restart supervisor (perintah tidak tersedia)."
fi

echo "[Setup] Selesai. Anda dapat mulai training:"
echo "  python run.py train --epochs 5 --batch-size 64 --output-dir ./models --model-name fashion_mnist_mlp"