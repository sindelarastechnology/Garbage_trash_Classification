# 🗑️ Garbage Classification — UAS Pengolahan Citra Digital

**Klasifikasi Sampah Otomatis menggunakan CNN dengan Transfer Learning (MobileNetV2)**

| | |
|---|---|
| **Mata Kuliah** | Pengolahan Citra Digital / CS20-3403 (3 SKS) |
| **Program Studi** | Ilmu Komputer |
| **Dosen** | Abd. Chairul Fauzan, M.Kom. |
| **Algoritma** | CNN + MobileNetV2 (Transfer Learning) |
| **Dataset** | [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) (Kaggle) |
| **Akurasi** | **91.79%** (Validation Accuracy) |

---

## 📋 Daftar Isi

- [Ringkasan Project](#ringkasan-project)
- [Folder Structure](#folder-structure)
- [Prerequisites / Dependencies](#prerequisites--dependencies)
- [Cara Clone & Setup](#cara-clone--setup)
- [Cara Download Dataset](#cara-download-dataset)
- [Cara Menjalankan](#cara-menjalankan)
- [Hasil & Evaluasi](#hasil--evaluasi)
- [Cara Menggunakan di VSCode](#cara-menggunakan-di-vscode)
- [Troubleshooting](#troubleshooting)
- [Lisensi](#lisensi)

---

## Ringkasan Project

Project ini dibuat untuk **Ujian Akhir Semester (UAS)** mata kuliah Pengolahan Citra Digital. Tujuannya adalah mengklasifikasikan gambar sampah ke dalam **6 kategori**:

| Kelas | Indonesia | Jumlah Data |
|-------|-----------|-------------|
| `cardboard` | Kardus | 2.214 |
| `glass` | Kaca | 2.500 |
| `metal` | Logam | 2.084 |
| `paper` | Kertas | 2.315 |
| `plastic` | Plastik | 2.288 |
| `trash` | Sampah Organik | 2.500 |
| **TOTAL** | | **13.901** |

### Metode yang Digunakan

1. **Preprocessing** — 7 teknik pengolahan citra (Resize, Grayscale, Normalisasi, Gaussian Blur, Otsu Segmentasi, Canny Edge Detection, Ekstraksi Histogram)
2. **Data Augmentation** — Rotation, shift, shear, zoom, flip untuk meningkatkan variasi data
3. **CNN + Transfer Learning** — MobileNetV2 (pre-trained ImageNet) dengan custom classification head
4. **Evaluasi** — Classification report, Confusion Matrix, uji prediksi gambar baru

### Hasil Utama

| Metrik | Nilai |
|--------|-------|
| Validation Accuracy | **91.79%** |
| Validation Loss | 0.2597 |
| F1-Score (weighted avg) | **0.92** |
| Training Accuracy | 92.03% |

---

## Folder Structure

```
Garbage_trash/
├── content/                          # Model yang sudah dilatih (.h5 + .keras)
│   ├── garbage_classifier_mobilenetv2.h5
│   └── garbage_classifier_mobilenetv2.keras
├── Test/                             # Gambar uji + notebook testing
│   ├── Test_Model.ipynb
│   ├── images.jpg, images2.jpg, ...  # 10 gambar uji
│
├── Trash_Dataset/                    # ⚠️ TIDAK DIUPLOAD (download sendiri)
│   ├── images/                       #   Folder berisi 6 sub-folder kelas
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── metadata.csv
│
├── UAS_Pengolahan_Citra_Garbage_Classification.ipynb  # Notebook utama
├── index.html                        # Panduan lengkap (buka di browser)
├── hasil_analisis.md                 # Analisis detail hasil
├── soal.md                           # Soal UAS
├── README.md                         # File ini
└── requirements.txt                  # Daftar dependencies Python
```

---

## Prerequisites / Dependencies

### A. Software yang Diperlukan

| Software | Keterangan |
|----------|-----------|
| **Python 3.8 - 3.10** | Jangan pakai Python 3.11+ (TensorFlow belum support penuh) |
| **VSCode** (opsional) | Editor rekomendasi, install ekstensi **Python** + **Jupyter** |
| **Git** | Untuk clone repository |

### B. Python Libraries

Semua dependencies ada di `requirements.txt`:

```txt
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.1.78
matplotlib==3.7.2
scikit-learn==1.3.0
seaborn==0.12.2
pillow==10.0.0
```

> **Catatan:** Versi di atas adalah versi yang sudah teruji. Jika ada konflik, install sesuai versi tersebut.

---

## Cara Clone & Setup

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/Garbage_trash.git
cd Garbage_trash
```

### 2. Buat Virtual Environment (Rekomendasi)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset (lihat instruksi di bawah)

---

## Cara Download Dataset

Dataset **tidak termasuk** dalam repository karena ukurannya besar. Ikuti langkah berikut:

### Opsi 1: Download dari Kaggle (Rekomendasi)

1. Buka: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
2. Klik **Download**
3. Extract file zip
4. Pindahkan folder `images/` dan `metadata.csv` ke dalam `Trash_Dataset/`
5. Pastikan struktur folder menjadi:

```
Trash_Dataset/
├── images/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── metadata.csv
```

### Opsi 2: Via Kaggle API

```bash
pip install kaggle
kaggle datasets download asdasdasasdas/garbage-classification
unzip garbage-classification.zip -d Trash_Dataset/
```

---

## Cara Menjalankan

### A. Buka Notebook di VSCode

```bash
code .
```

1. Buka file `UAS_Pengolahan_Citra_Garbage_Classification.ipynb`
2. Pilih kernel Python dari virtual environment (`venv`)
3. Jalankan cell secara berurutan (**Run All** atau satu per satu)

### B. Buka Panduan HTML

Buka `index.html` di browser untuk melihat panduan lengkap, penjelasan kode, dan skrip video.

### C. Testing Model (Optional)

Buka `Test/Test_Model.ipynb` untuk menguji model pada gambar baru. Ganti `TEST_IMAGE` di cell kode dengan path gambar yang ingin diuji.

---

## Hasil & Evaluasi

### Classification Report

| Kelas | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| cardboard | 0.94 | 0.93 | 0.94 |
| glass | 0.93 | 0.89 | 0.91 |
| metal | 0.88 | 0.97 | 0.93 |
| paper | 0.91 | 0.95 | 0.93 |
| plastic | 0.91 | 0.91 | 0.91 |
| trash | 0.94 | 0.86 | 0.90 |
| **Weighted Avg** | **0.92** | **0.92** | **0.92** |

### Kesimpulan

✅ **Model BERHASIL** melakukan pengolahan citra dengan baik.
Akurasi 91.79% menunjukkan model mampu membedakan jenis sampah dengan sangat baik.

---

## Cara Menggunakan di VSCode

### Ekstensi yang Perlu Diinstall

| Ekstensi | Fungsi |
|----------|--------|
| **Python** (Microsoft) | IntelliSense, debugging, linting |
| **Jupyter** (Microsoft) | Menjalankan notebook .ipynb |
| **GitLens** (opsional) | Melihat history git di editor |

### Langkah-langkah

1. **Install VSCode** dari [code.visualstudio.com](https://code.visualstudio.com/)
2. **Install ekstensi** Python + Jupyter dari tab Extensions (Ctrl+Shift+X)
3. **Clone repositori** (lihat cara di atas)
4. **Buka folder**: `File → Open Folder → pilih Garbage_trash`
5. **Buat virtual environment**:
   - Buka terminal: `` Ctrl+` ``
   - Jalankan: `python -m venv venv`
   - Aktifkan: `venv\Scripts\activate` (Windows) / `source venv/bin/activate` (Mac/Linux)
6. **Install dependencies**: `pip install -r requirements.txt`
7. **Download dataset** dan letakkan di `Trash_Dataset/`
8. **Buka notebook**: Klik file `UAS_Pengolahan_Citra_Garbage_Classification.ipynb`
9. **Pilih kernel**: Klik "Select Kernel" di kanan atas → pilih Python dari `venv`
10. **Jalankan**: Klik "Run All" atau jalankan cell satu per satu dengan Shift+Enter

### Tips VSCode

- **Run cell**: `Shift+Enter`
- **Run all**: Klik `Run All` di toolbar notebook
- **Clear output**: Klik kanan cell → `Clear Output`
- **Restart kernel**: Klik tombol restart di toolbar atau `Ctrl+Shift+P` → `Jupyter: Restart Kernel`

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| **TensorFlow tidak terdeteksi** | Install ulang: `pip install tensorflow==2.13.0` |
| **GPU Warning** | Warning "GPU tidak didukung di Windows" aman diabaikan. Training tetap jalan di CPU (lebih lambat). |
| **ModuleNotFoundError** | Jalankan `pip install -r requirements.txt` |
| **FileNotFoundError: Trash_Dataset** | Dataset belum didownload. Lihat bagian [Cara Download Dataset](#cara-download-dataset). |
| **Kernel error di VSCode** | Restart kernel: `Ctrl+Shift+P` → `Jupyter: Restart Kernel`. Pastikan kernel menggunakan venv. |
| **OpenCV error** | Install: `pip install opencv-python==4.8.1.78` |

---

## Lisensi

Project ini dibuat untuk keperluan akademis mata kuliah **Pengolahan Citra Digital (CS20-3403)**.

Dataset: [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) — lisensi sesuai ketentuan Kaggle.

---

<div align="center">
  <p><strong>UAS Pengolahan Citra Digital — Garbage Classification</strong></p>
  <p>CNN + MobileNetV2 Transfer Learning | Akurasi 91.79%</p>
</div>
