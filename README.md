# Mini-Vision System: Fashion-MNIST Classifier (MLP)

Pembuat: Lycus | Muhammad Affif

Ringkas: Proyek ini melatih model klasifikasi gambar sederhana (ANN/MLP) pada dataset Fashion-MNIST menggunakan TensorFlow/Keras, melakukan normalisasi piksel (0-1), evaluasi, dan ekspor model ke format Keras (.keras) dan TensorFlow Lite (.tflite) untuk dibawa ke aplikasi mobile/edge.

Fitur Utama
- Dataset: Fashion-MNIST (28x28 grayscale) – otomatis terunduh dari Keras
- Arsitektur: Flatten → Dense(512, ReLU) → Dense(10, Softmax)
- Training Default: epochs=5, batch_size=64
- Output Artefak: .keras, .tflite, dan metrics .json di folder models/

Struktur Repositori (relevan)
- scripts/train_fashion_mnist.py → Script utama training + ekspor TFLite
- scripts/setup_ml_env.sh → Setup cepat dependensi ML (opsional, sekali jalan)
- run.py → Satu pintu untuk setup/train/verify/paths
- backend/ → FastAPI service (tidak ditambah endpoint ML pada MVP ini)
- frontend/ → React app (tidak digunakan di MVP ini)

Persyaratan
- Python 3.11+
- Pip aktif pada environment (virtualenv/venv disarankan)
- Koneksi internet untuk unduh dataset Fashion-MNIST

Quick Start (Paling Cepat)
1) Setup (sekali jalan, instal TensorFlow dan sinkronisasi requirements)
   bash scripts/setup_ml_env.sh

2) Training (hasil simpan ke ./models)
   python run.py train --epochs 5 --batch-size 64 --output-dir ./models --model-name fashion_mnist_mlp

3) Verifikasi cepat (inference TFLite pada sampel test)
   python run.py verify --model ./models/fashion_mnist_mlp.tflite

4) Lihat path artefak terakhir (berdasarkan metrics)
   python run.py paths --dir ./models

Perintah Lengkap via run.py
- Setup environment ML:
  python run.py setup

- Training:
  python run.py train --epochs 5 --batch-size 64 --output-dir ./models --model-name fashion_mnist_mlp

- Verifikasi TFLite (inference cepat pada sampel dataset test):
  python run.py verify --model ./models/fashion_mnist_mlp.tflite

- Menampilkan path artefak/model terbaru dari metrics:
  python run.py paths --dir ./models

Detail Output
- models/<nama>.keras → Model Keras lengkap
- models/<nama>.tflite → Model TFLite siap mobile/edge
- models/<nama>_metrics.json → Metrik training (train/val accuracy, test accuracy, waktu training, dll)

Catatan Arsitektur & Lingkungan
- MVP ini tidak menambah endpoint API atau UI. Fokus: ML scripting + ekspor model.
- Backend FastAPI/Frontend sudah tersedia untuk pengembangan berikutnya, gunakan variabel lingkungan yang ada (JANGAN ubah .env).
- Semua route backend harus prefiks /api (aturan ingress), bila nanti menambah endpoint.

Troubleshooting
- TensorFlow belum terinstal / error import tensorflow:
  Jalankan: bash scripts/setup_ml_env.sh

- Backend restart setelah update dependencies:
  scripts/setup_ml_env.sh akan mencoba restart backend via supervisor. Jika gagal (izin sudo), jalankan manual:
  sudo supervisorctl restart backend

- Lambat saat training pertama: dataset diunduh sekali. Jalankan ulang training akan lebih cepat.

Lisensi
- Gunakan bebas untuk pembelajaran/praktik. Cantumkan kredit: Pembuat — Lycus.