# 🧠 Mini-Vision System: Fashion-MNIST Classifier (MLP)  
**Pembuat**: Lycus | Muhammad Affif  
**Proyek oleh**: [Nourivex Tech](https://nourivex.tech)  

> Sistem klasifikasi gambar sederhana berbasis *Multi-Layer Perceptron* (MLP) pada dataset **Fashion-MNIST**, dirancang untuk pelatihan, evaluasi, dan ekspor model ke format **TensorFlow Lite** — siap digunakan di edge device atau aplikasi mobile.  
> Proyek ini merupakan fondasi MVP dari ekosistem **Lycus Coder**, dengan arsitektur modular yang mendukung pengembangan backend (FastAPI) dan frontend (React) di masa depan.

---

## 📌 Ringkasan

Proyek ini:
- Melatih model ANN/MLP menggunakan **TensorFlow/Keras**
- Melakukan normalisasi piksel (0–1) dan evaluasi akurasi
- Mengekspor model ke format `.keras` dan `.tflite`
- Menyediakan skrip terpusat (`run.py`) untuk pelatihan, verifikasi, dan manajemen artefak
- Dirancang dengan struktur **full-stack-ready**: backend (FastAPI) dan frontend (React) sudah tersedia untuk integrasi lanjutan

---

## ✨ Fitur Utama

- **Dataset**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (28×28 grayscale) — otomatis diunduh via Keras
- **Arsitektur Model**:  
  `Flatten → Dense(512, ReLU) → Dense(10, Softmax)`
- **Training Default**: `epochs=5`, `batch_size=64`
- **Output Artefak** (disimpan di `./models/`):
  - `.keras` → Model lengkap TensorFlow/Keras
  - `.tflite` → Model ringan untuk edge/mobile
  - `_metrics.json` → Akurasi, waktu training, dan metadata evaluasi

---

## 🗂️ Struktur Repositori (Relevan)

```
.
├── backend/          # FastAPI service (siap untuk endpoint ML di iterasi berikutnya)
├── frontend/         # React + Tailwind UI (komponen siap pakai, belum terhubung ke model)
├── models/           # Output model & metrik
├── scripts/
│   ├── train_fashion_mnist.py   # Logika pelatihan & ekspor TFLite
│   └── setup_ml_env.sh          # Instalasi dependensi ML
├── run.py            # CLI terpusat: setup, train, verify, paths
├── README.md
└── gambar/           # Screenshot UI (contoh: tampilan_frontend.png)
```

> 💡 **Catatan Arsitektur**:  
> - MVP ini **tidak mengaktifkan endpoint ML di backend** atau **koneksi ke frontend**.  
> - Namun, struktur sudah disiapkan untuk integrasi penuh:  
>   - Backend menggunakan rute `/api/*` (sesuai aturan ingress)  
>   - Frontend menggunakan komponen UI modern (shadcn-style + Tailwind)  
>   - Gunakan `.env` yang tersedia — **jangan ubah langsung**.

---

## ⚙️ Persyaratan

- Python 3.11+
- `pip` aktif (disarankan dalam `venv` atau `conda`)
- Koneksi internet (untuk unduh dataset pertama kali)
- (Opsional) Node.js + Yarn (jika ingin menjalankan frontend)

---

## 🚀 Quick Start (Paling Cepat)

### 1. Setup Lingkungan ML (sekali jalan)
```bash
bash scripts/setup_ml_env.sh
```

### 2. Latih Model & Ekspor ke TFLite
```bash
python run.py train --epochs 5 --batch-size 64 --output-dir ./models --model-name fashion_mnist_mlp
```

### 3. Verifikasi Model TFLite (inference pada sampel acak)
```bash
python run.py verify --model ./models/fashion_mnist_mlp.tflite
```

### 4. Lihat Path Artefak Terbaru
```bash
python run.py paths --dir ./models
```

---

## 🛠️ Perintah Lengkap via `run.py`

| Perintah | Deskripsi |
|--------|----------|
| `python run.py setup` | Instal dependensi ML (TensorFlow, dll) |
| `python run.py train [...]` | Latih model dengan parameter kustom |
| `python run.py verify --model <path>` | Uji inferensi TFLite pada data test |
| `python run.py paths --dir <folder>` | Tampilkan model & metrik terbaru berdasarkan timestamp |

---

## 📤 Detail Output

Setelah pelatihan, folder `models/` akan berisi:
- `fashion_mnist_mlp.keras` → Model Keras penuh (untuk fine-tuning)
- `fashion_mnist_mlp.tflite` → Model terkuantisasi (siap deploy ke mobile/edge)
- `fashion_mnist_mlp_metrics.json` → Berisi:
  ```json
  {
    "train_accuracy": 0.892,
    "val_accuracy": 0.875,
    "test_accuracy": 0.871,
    "training_time_sec": 42.3,
    "epochs": 5,
    "batch_size": 64,
    "timestamp": "2025-04-05T14:30:00Z"
  }
  ```

---

## 🖥️ Frontend & Backend (Persiapan untuk Iterasi Berikutnya)

- **Frontend** (`/frontend`):  
  Dibangun dengan **React + Tailwind CSS**, menggunakan komponen UI modular (shadcn-inspired).  
  Screenshot UI tersedia di `gambar/tampilan_frontend.png`.

- **Backend** (`/backend`):  
  Server FastAPI siap menerima endpoint seperti:
  ```python
  @app.post("/api/predict")
  async def predict(image: UploadFile):
      # ... inference TFLite ...
  ```
  Semua rute **harus** menggunakan prefiks `/api` (sesuai kebijakan ingress Nourivex).

> 🔜 **Rencana Pengembangan**:  
> Integrasi frontend ↔ backend ↔ model TFLite untuk demo klasifikasi gambar interaktif.

---

## 🛑 Troubleshooting

| Masalah | Solusi |
|-------|--------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Jalankan `bash scripts/setup_ml_env.sh` |
| Backend tidak restart otomatis setelah setup | Jalankan manual: `sudo supervisorctl restart backend` |
| Training lambat di run pertama | Normal — dataset diunduh sekali. Run kedua jauh lebih cepat. |
| Error saat verifikasi TFLite | Pastikan path file `.tflite` benar dan tidak rusak |

---

## 📜 Lisensi

Proyek ini **bebas digunakan untuk pembelajaran, eksperimen, atau pengembangan internal**.  
Jika digunakan atau dikembangkan lebih lanjut, mohon cantumkan kredit:

> **Pembuat**: Lycus | Muhammad Affif  
> **Proyek**: Mini-Vision System — Lycus Coder by Nourivex Tech

---

> 🌐 **Dikembangkan dengan ❤️ di Nourivex Tech**  
> [https://nourivex.tech](https://nourivex.tech)

