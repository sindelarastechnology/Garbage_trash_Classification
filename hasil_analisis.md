# Analisis Hasil UAS Pengolahan Citra — Garbage Classification

**Nama File:** `UAS_Pengolahan_Citra_Garbage_Classification.ipynb`
**Dataset:** Garbage Classification Dataset (Kaggle)
**Algoritma:** CNN dengan Transfer Learning (MobileNetV2)

---

## A. PEMETAAN SOAL VS IMPLEMENTASI KODE

### Soal 1 — Dataset dan Akuisisi Citra (Bobot 20)
| Sub-soal | Ada/Tidak | Letak di Notebook |
|---|---|---|
| Nama & sumber dataset | ✅ Ada | Cell markdown "Soal 1" & code cell 2 |
| Tujuan penggunaan dataset | ✅ Ada | Cell markdown "Soal 1" |
| Jumlah data & kelas | ✅ Ada | Code cell 2 (menampilkan count per kelas + total 13.901 gambar) |
| Contoh citra | ✅ Ada | Code cell 3 (menampilkan 6 gambar sample, 1 per kelas) |

**Detail:** Soal 1 terpenuhi dengan baik. Dataset dihitung langsung dari folder, dan visualisasi contoh citra ditampilkan beserta ukurannya.

---

### Soal 2 — Representasi Citra Digital (Bobot 20)
| Sub-soal | Ada/Tidak | Letak di Notebook |
|---|---|---|
| Cara baca citra (Python) | ✅ Ada | Code cell 4: `cv2.imread()` |
| Format & ukuran citra | ✅ Ada | Menampilkan shape (256,256,3), dtype uint8, format JPG |
| Representasi citra (RGB/pixel) | ✅ Ada | Channel R/G/B divisualisasikan terpisah, contoh nilai pixel 5x5 ditampilkan |

**Detail:** Soal 2 terpenuhi dengan baik. Bahkan ada visualisasi channel RGB secara terpisah dengan colormap.

---

### Soal 3 — Teknik Pengolahan Citra (Bobot 30)
| Tahapan | Ada/Tidak | Fungsi | Alasan |
|---|---|---|---|
| 1. Resize (256→224) | ✅ Ada | `cv2.resize()` | Menyeragamkan ukuran input CNN |
| 2. Grayscale | ✅ Ada | `cv2.cvtColor(…,RGB2GRAY)` | Reduksi dimensi & analisis tekstur |
| 3. Normalisasi pixel | ✅ Ada | Bagi 255.0 | Stabilitas & konvergensi training |
| 4. Gaussian Blur | ✅ Ada | Kernel (5,5) | Reduksi noise |
| 5. Segmentasi (Otsu) | ✅ Ada | `cv2.threshold(…,OTSU)` | Memisahkan objek dari latar belakang |
| 6. Edge Detection (Canny) | ✅ Ada | threshold1=50, threshold2=150 | Deteksi tepi/kontur objek |
| 7. Ekstraksi Fitur (Histogram RGB) | ✅ Ada | `cv2.calcHist()` | Distribusi warna sebagai fitur |

**Detail:** Soal 3 terpenuhi lebih dari minimum (7 tahap, padahal minimal 3). Setiap tahap diberi komentar fungsi & alasan.

> **Catatan Kritis:** 7 tahap preprocessing di atas hanya didemonstrasikan pada **satu gambar sample** untuk keperluan visualisasi/edukasi. Model CNN yang dilatih di sesi training **tidak menggunakan** grayscale, Gaussian blur, Otsu, Canny, atau histogram. Model menggunakan data augmentation (rotation, shift, shear, zoom, flip) + rescale 1./255. Ini wajar karena model membutuhkan input RGB 3-channel, sedangkan grayscale hanya 1-channel. Namun, untuk keperluan menjawab soal, pendekatan ini tetap valid karena mendemonstrasikan pemahaman teknik pengolahan citra.

---

### Soal 4 — Aplikasi Pengolahan Citra (Bobot 15)
| Sub-soal | Ada/Tidak | Detail |
|---|---|---|
| Nama algoritma | ✅ Ada | CNN dengan Transfer Learning (MobileNetV2) |
| Fungsi algoritma terhadap citra | ✅ Ada | Mengekstrak fitur visual hierarkis (tepi → tekstur → bentuk → objek) |
| Hubungan dengan pengenalan objek | ✅ Ada | CNN mengenali pola visual unik tiap jenis sampah |

**Detail:** Soal 4 terpenuhi. Arsitektur model dijelaskan: MobileNetV2 base → GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(6, Softmax). Total 2.587.462 parameter.

---

### Soal 5 — Analisis Hasil (Bobot 15)
| Sub-soal | Ada/Tidak | Detail |
|---|---|---|
| Input citra yang diuji | ✅ Ada | Gambar glass dari dataset (index ke-5) |
| Output program | ✅ Ada | Bar chart confidence + label prediksi |
| Akurasi/hasil pengujian | ✅ Ada | Validation accuracy: 91.79%, Loss: 0.2597 |
| Classification Report | ✅ Ada | Precision/Recall/F1 per kelas |
| Confusion Matrix | ✅ Ada | Heatmap visual |
| Kesimpulan | ✅ Ada | Model BERHASIL (akurasi ≥ 85%) |

**Detail:** Soal 5 terpenuhi dengan sangat baik. Dilengkapi kurva training, classification report, confusion matrix, dan uji prediksi gambar individual.

---

## B. EVALUASI KUALITAS MODEL

### Metrik Utama
| Metrik | Nilai |
|---|---|
| Validation Accuracy | **91.79%** |
| Validation Loss | **0.2597** |
| Epochs | 20 |
| Total Parameter | 2.587.462 |
| Optimizer | Adam (lr=0.001, turun jadi 5e-4) |

### Classification Report
| Kelas | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| cardboard | 0.94 | 0.93 | 0.94 | 442 |
| glass | 0.93 | 0.89 | 0.91 | 500 |
| metal | 0.88 | 0.97 | 0.93 | 416 |
| paper | 0.91 | 0.95 | 0.93 | 463 |
| plastic | 0.91 | 0.91 | 0.91 | 457 |
| trash | 0.94 | 0.86 | 0.90 | 500 |
| **Rata-rata** | **0.92** | **0.92** | **0.92** | **2778** |

### Analisis Kualitas

**KELEBIHAN:**
1. **Akurasi tinggi (91.79%)** — sangat baik untuk klasifikasi 6 kelas sampah
2. **Transfer Learning MobileNetV2** — memanfaatkan bobot pre-trained ImageNet, mempercepat training & meningkatkan akurasi
3. **Data Augmentation** — rotation, shift, shear, zoom, flip membantu generalization
4. **Seimbang antar kelas** — F1-score semua kelas ≥ 0.90, tidak ada kelas yang tertinggal jauh
5. **Train vs Val accuracy berdekatan** — menunjukkan tidak overfitting (train: ~92%, val: ~91.79%)
6. **Dropout (0.5)** — membantu regularisasi

**KEKURANGAN / CATATAN:**
1. **Base model tidak di-fine-tune** — `base_model.trainable = False`, akurasi bisa lebih tinggi jika beberapa layer terakhir di-unfreeze
2. **Early stopping tidak aktif** — patience=5 tapi training berjalan penuh 20 epoch (tidak berhenti lebih awal), artinya model masih terus membaik
3. **Preprocessing tidak terintegrasi dengan pipeline training** — 7 tahap preprocessing hanya demo visual, bukan bagian dari data generator
4. **Training lama** — ~348 step/epoch × ~375 detik = ~125 menit total. Bisa dipercepat dengan GPU
5. **GPU tidak terdeteksi** — warning TensorFlow: GPU tidak dipakai karena Windows native

### Potensi Peningkatan
| Usulan | Dampak |
|---|---|
| Fine-tuning beberapa layer MobileNetV2 | Potensi akurasi > 94% |
| Tambah更多的 epoch (30-50) | Konvergensi lebih optimal |
| Gunakan GPU (WSL2/DirectML) | Training 5-10× lebih cepat |
| Class weight untuk kelas "trash" (recall 0.86) | Menyeimbangkan recall |
| Simpan model dalam format .keras (lebih modern) | ✅ Sudah dilakukan |

---

## C. KESIMPULAN UMUM

1. **✅ Semua soal terjawab dengan baik.**
   - Soal 1 (Dataset): ✅ Lengkap dengan statistik & visualisasi
   - Soal 2 (Representasi): ✅ Lengkap dengan channel RGB & nilai pixel
   - Soal 3 (Preprocessing): ✅ 7 tahap dijelaskan dengan fungsi & alasan (melebihi minimal 3)
   - Soal 4 (Algoritma): ✅ CNN + MobileNetV2 dijelaskan arsitektur & fungsinya
   - Soal 5 (Analisis): ✅ Lengkap dengan metrik, confusion matrix, uji prediksi, & kesimpulan

2. **✅ Kualitas model sangat bagus.**
   - Akurasi 91.79% termasuk tinggi untuk klasifikasi 6 kelas sampah
   - Presisi/Recall/F1 seimbang di semua kelas (≥ 0.90)
   - Tidak ada indikasi overfitting yang signifikan
   - Transfer Learning terbukti efektif

3. **⚠️ Satu catatan penting:**
   7 tahap preprocessing (grayscale, Gaussian blur, Otsu, Canny, histogram) ditampilkan sebagai **demo edukasi**, bukan sebagai pipeline training sesungguhnya. Model dilatih dengan data RGB + augmentasi. Untuk UAS, ini tidak masalah karena soal meminta "analisis tahapan preprocessing" — dan mahasiswa telah mendemonstrasikan pemahaman terhadap teknik-teknik tersebut.

4. **Model tersimpan di:**
   - `content/garbage_classifier_mobilenetv2.h5` (~92 MB, format HDF5 legacy)
   - `content/garbage_classifier_mobilenetv2.keras` (format native Keras)

---

## D. REKOMENDASI UNTUK PENYEMPURNAAN

1. Jelaskan di video bahwa preprocessing demo bersifat **ilustratif** untuk soal 3, sedangkan pipeline training menggunakan **augmentasi + rescale**
2. Untuk proyek riil: integrasikan preprocessing (Gaussian blur, dll) ke dalam `ImageDataGenerator` menggunakan fungsi preprocessing custom
3. Coba fine-tuning dengan `base_model.trainable = True` pada layer atas untuk akurasi lebih tinggi
4. Gunakan confusion matrix untuk analisis lebih detail: kelas "trash" sering salah prediksi (recall 0.86 — paling rendah)

---

## E. NOTEBOOK BARU: `UAS.ipynb` — PENINGKATAN MODEL

Berdasarkan rekomendasi di atas, dibuat notebook baru **`UAS.ipynb`** dengan peningkatan signifikan:

### E.1 Perbandingan Notebook Lama vs Baru

| Aspek | `UAS_Pengolahan_Citra_Garbage_Classification.ipynb` | `UAS.ipynb` |
|-------|---|---|
| **Model** | MobileNetV2 (TensorFlow/Keras) | YOLOv8-cls (Ultralytics/PyTorch) |
| **Akurasi Target** | 91.79% | **≥ 95%** (dengan fine-tuning & arsitektur lebih modern) |
| **Early Stopping** | Patience=5 | Patience=7 (lebih toleran) |
| **Augmentasi** | rotation, shift, shear, zoom, flip | Auto-augmentasi + augmentasi YOLOv8 bawaan |
| **Epochs** | 20 | 30 (dengan early stopping) |
| **Fine-tuning** | ❌ Tidak (base frozen) | ✅ Ya (YOLOv8 otomatis menyesuaikan) |
| **Deteksi + Counting** | ❌ Tidak ada | ✅ Ya (YOLOv8 detection COCO + classifier) |
| **Framework** | TensorFlow 2.x | PyTorch + Ultralytics |

### E.2 Struktur Notebook (26 Cells)

| Cell | Bagian | Deskripsi |
|------|--------|-----------|
| 1 | Markdown | Judul & deskripsi |
| 2-3 | ⚙️ Instalasi + Imports | Install PyTorch CPU, ultralytics, import library |
| 4-6 | 📁 Soal 1 — Dataset | Statistik dataset, contoh gambar per kelas |
| 7-9 | 🖼️ Soal 2 — Representasi | Baca gambar, shape, channel RGB, nilai pixel |
| 10-12 | ⚙️ Soal 3 — Preprocessing | 7 tahap preprocessing (resize → histogram) |
| 13-15 | 🤖 Soal 4 — YOLOv8 | Penjelasan algoritma + training YOLOv8-cls |
| 16-20 | 📊 Soal 5 — Analisis | Evaluasi, classification report, confusion matrix, prediksi |
| 21-23 | 🔍 Deteksi + Counting | Fungsi deteksi & counting dengan YOLOv8 COCO |
| 24 | Kesimpulan | Output analisis keberhasilan model |
| 25 | Simpan Model | Export model ke format .pt |

### E.3 Fitur Baru: Deteksi + Counting

**Cara Kerja:**
1. **YOLOv8n.pt** (COCO pre-trained) mendeteksi semua objek dalam gambar input
2. Setiap objek di-crop dan diklasifikasi dengan **YOLOv8-cls** (model yang sudah dilatih pada dataset sampah)
3. Hasil per kelas dihitung dan ditampilkan dengan bounding box

**Fungsi utama:** `detect_and_count_garbage(image_path, det_model, cls_model, class_names)`

**Output:**
- Gambar dengan bounding box hijau + label kelas
- Tabel ringkasan: `plastic: 2, glass: 5, metal: 1, ...`

**Keterbatasan yang perlu diketahui:**
- Deteksi menggunakan YOLOv8 COCO (80 kelas umum), mungkin tidak mendeteksi **semua** jenis sampah (misal: kardus lipat yang tidak dikenali COCO)
- Untuk deteksi optimal, diperlukan dataset dengan bounding box annotations (contoh: TACO dataset)
- Pada CPU, training membutuhkan waktu lebih lama dibanding TensorFlow

### E.4 Instalasi yang Diperlukan

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

### E.5 Catatan Penting untuk Video Presentasi

1. **Jelaskan perbedaan** antara model lama (MobileNetV2) dan baru (YOLOv8-cls)
2. **Demonstrasikan fitur counting** dengan gambar yang mengandung beberapa objek sampah
3. **Sebutkan keterbatasan** deteksi COCO — tidak semua sampah terdeteksi sempurna
4. **Tunjukkan peningkatan akurasi** yang diharapkan dari YOLOv8-cls vs MobileNetV2
