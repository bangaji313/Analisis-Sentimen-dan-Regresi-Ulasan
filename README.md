# Proyek Machine Learning: Analisis Sentimen dan Prediksi Skor Ulasan Aplikasi

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)

Proyek ini merupakan tugas besar untuk mata kuliah IFB-310 Machine Learning. Tujuannya adalah untuk membangun dan mengevaluasi serangkaian model *machine learning* untuk menganalisis ulasan pengguna aplikasi **"Wondr by BNI"** dari Google Play Store.

Proyek ini memiliki dua tujuan utama:
1.  **Analisis Sentimen (Klasifikasi):** Mengkategorikan ulasan ke dalam sentimen **positif, netral, atau negatif**.
2.  **Prediksi Skor (Regresi):** Memprediksi skor numerik (rating **1-5**) berdasarkan isi teks ulasan.

Seluruh proses pengembangan proyek ini mengikuti metodologi **CRISP-DM**, dan hasilnya diimplementasikan dalam sebuah aplikasi web interaktif menggunakan Streamlit.

---

### **Fitur Utama**
- **Klasifikasi Sentimen:** Menggunakan model **Logistic Regression** untuk mengidentifikasi opini pengguna.
- **Prediksi Skor:** Menggunakan model **Random Forest Regressor** untuk memberikan skor prediktif yang lebih bernuansa.
- **Aplikasi Web Interaktif:** Antarmuka sederhana yang dibangun dengan Streamlit untuk pengujian model secara *real-time*.
- **Perbandingan 10 Model:** Melakukan eksperimen dengan 5 model klasifikasi dan 5 model regresi untuk menemukan performa terbaik.

### **Metodologi: CRISP-DM**
Proyek ini dikembangkan dengan mengikuti 6 fase dari kerangka kerja CRISP-DM:
1.  **Business Understanding:** Memahami kebutuhan BNI untuk menganalisis *feedback* pengguna secara efisien guna meningkatkan retensi dan kepuasan pelanggan.
2.  **Data Understanding:** Menganalisis data ulasan mentah dari Play Store, mengidentifikasi distribusi skor yang tidak seimbang dan kata kunci utama melalui EDA.
3.  **Data Preparation:** Membersihkan dan mentransformasi data teks melalui normalisasi, tokenisasi, *stopword removal*, dan *stemming*.
4.  **Modeling:** Melatih 10 model machine learning berbeda dengan variasi fitur dan rasio data untuk tugas klasifikasi dan regresi.
5.  **Evaluation:** Mengevaluasi performa model menggunakan metrik yang relevan (F1-Score, Presisi, MAE, R²) untuk memilih model terbaik yang paling sesuai dengan tujuan bisnis.
6.  **Deployment:** Mengimplementasikan dua model terbaik ke dalam sebuah prototipe aplikasi web interaktif sebagai bukti konsep.

### **Teknologi yang Digunakan**
- **Bahasa:** Python 3.9
- **Framework Aplikasi:** Streamlit
- **Analisis & ML:** Pandas, NumPy, Scikit-learn, XGBoost
- **NLP:** Sastrawi (Stemming), Gensim (Word2Vec)
- **Visualisasi:** Matplotlib, Seaborn, WordCloud

### **Hasil Akhir**
Setelah melakukan perbandingan, model terbaik untuk setiap tugas adalah:

**Model Klasifikasi Terbaik**
| Model | Akurasi | Presisi (Macro) | Recall (Macro) | F1-Score (Macro)|
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression + TF-IDF** | 86.10% | 0.6832 | 0.5834 | 0.5750 |

**Model Regresi Terbaik**
| Model | MAE | RMSE | R-squared (R²) |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor + TF-IDF** | 0.5595 | 1.0131 | 0.6159 |

### **Cara Menjalankan Aplikasi**

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/bangaji313/Analisis-Sentimen-dan-Regresi-Ulasan.git
    cd Analisis-Sentimen-dan-Regresi-Ulasan
    ```

2.  **(Opsional) Buat dan aktifkan virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan terbuka secara otomatis di *browser* Anda.

---

### **Tim Proyek**
Proyek ini dikerjakan secara berkelompok oleh:
- Muhammad Figo Razzan Fadilah (152022064)
- Maulana Seno Aji Yudhantara (152022065)
- Naizirun De Jesus Da Silva (152022077)

### **Konteks Akademik**
Proyek ini merupakan bagian dari tugas besar mata kuliah **IFB-310 Machine Learning** di Program Studi Informatika, Fakultas Teknologi Industri, **Institut Teknologi Nasional Bandung**. Dosen Pengampu: Bapak Fahmi Arif, S.T., M.T., Ph.D.
