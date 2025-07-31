# Optical-Character-Recognition
Optical Character Recognition (OCR) adalah teknologi yang memungkinkan komputer untuk mengenali dan mengkonversi teks yang terdapat dalam gambar

Proyek ini bertujuan untuk melakukan *Optical Character Recognition (OCR)* pada gambar plat nomor kendaraan menggunakan *Visual Language Model (VLM)* yang dijalankan melalui **LMStudio API** secara lokal. Evaluasi dilakukan dengan menghitung *Character Error Rate (CER)* untuk menilai seberapa akurat hasil pembacaan model

Tujuan 
* Menerapkan **VLM "llava-v1.5-7b-gpt4ocr-hf"** untuk mengenali karakter dari gambar plat nomor
* Membandingkan hasil prediksi dengan label asli (*ground truth*) menggunakan metrik **CER**
* Menghasilkan file CSV berisi ringkasan prediksi dan evaluasi

Structure Folder 

├── main.py                 Main application file
├── requirements.txt        Python dependencies
├── README.md               This file
├── log.txt                Processing logs (generated)
├── ocr_results.csv       Results output (generated)
└── dataset/              Your image dataset
    ├── images/
    └── ground_truth.csv

# Format File ground_truth.csv
csv
image,ground_truth
plat1.jpg,B1234XYZ
plat2.jpg,D5678ABC

Kolom `image` harus sesuai nama file di folder `test/
Kolom `ground_truth` berisi teks plat nomor asli.

# Prasyarat

- Python 3.8 atau lebih tinggi
- LMStudio terinstal dan berjalan di "http://localhost:1234"
- Model VLM (misalnya `"llava-v1.5-7b-gpt4ocr-hf"**) aktif di LMStudio


# Cara Menjalankan

Letakkan gambar dalam folder `test/`
Siapkan file `ground_truth.csv`
Jalankan script:

```bash
python UAS.py

# program setelah berjalan 

Gambar dikonversi ke base64 dan dikirim ke LMStudio
Model VLM memproses dan memprediksi teks plat nomor
Prediksi dibandingkan dengan `ground_truth` → dihitung nilai **CER**
Hasil dicetak ke terminal dan disimpan ke `ocr_results.csv`

Contoh Hasil Output
# Di Terminal:

Processing: plat1.jpg
Ground Truth: B1234XYZ
Prediction: B1234XYZ
CER Score: 0.000


# Di File `ocr_results.csv`

csv
image,ground_truth,prediction,CER_score
plat1.jpg,B1234XYZ,B1234XYZ,0.000
plat2.jpg,D5678ABC,D5678AC,0.1423

**Teknologi yang Digunakan**
| Library        | Fungsi                                                        |
| -------------- | ------------------------------------------------------------- |
| `os`           | Mengelola path dan file pada sistem operasi                   |
| `csv`          | Membaca dan menulis file CSV                                  |
| `base64`       | Encode gambar ke base64 untuk dikirim via API                 |
| `requests`     | Mengirim HTTP request ke LM Studio API                        |
| `difflib`      | Menghitung **Character Error Rate (CER)**                     |
| `re`           | Membersihkan dan memvalidasi prediksi plat nomor dengan regex |
| `time`         | Menambahkan delay antar request                               |
| `Pillow (PIL)` | Menangani gambar (opsional, tapi diimpor)                     |
| `dataclasses`  | Representasi hasil OCR dalam bentuk `@dataclass`              |
| `typing`       | Type hinting (List, Dict, dsb)                                |
| `pathlib`      | Menangani path dengan cara modern berbasis objek              |




