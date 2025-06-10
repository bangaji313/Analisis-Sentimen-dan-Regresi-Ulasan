import streamlit as st
import pandas as pd
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

# --- KONFIGURASI HALAMAN (Harus menjadi perintah st pertama) ---
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊", layout="wide")


# --- FUNGSI PREPROCESSING & LOAD MODEL (Menggunakan cache agar lebih cepat) ---

@st.cache_data
def load_stopwords():
    stopword_df = pd.read_csv('stopwordbahasa.csv', header=None, names=['stopwords'])
    return set(stopword_df['stopwords'].values)

@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_models():
    try:
        model_klasifikasi = joblib.load('model_logreg_tfidf.h5')
        model_regresi = joblib.load('model_rfreg_tfidf.h5')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model_klasifikasi, model_regresi, tfidf_vectorizer
    except FileNotFoundError:
        st.error("File model atau vectorizer tidak ditemukan. Pastikan semua file berada di folder yang sama dengan `app.py`.")
        return None, None, None

# Muat semua komponen yang dibutuhkan
stopwords_id = load_stopwords()
stemmer = get_stemmer()
model_klasifikasi, model_regresi, tfidf_vectorizer = load_models()


# Fungsi-fungsi preprocessing
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_text(text):
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords_id]

def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]


# --- TAMPILAN UTAMA APLIKASI WEB (UI) ---

st.title("📊 Analisis Sentimen Ulasan Aplikasi")
st.markdown("Aplikasi ini memprediksi sentimen (Positif/Negatif/Netral) dan skor (1-5) dari sebuah ulasan.")

# Area input teks dari pengguna
input_review = st.text_area("Masukkan teks ulasan di sini:", height=150, placeholder="Contoh: Aplikasinya bagus dan mudah digunakan!")

# Tombol untuk melakukan prediksi
if st.button("Analisis Sentimen"):
    if all([model_klasifikasi, model_regresi, tfidf_vectorizer]) and input_review:
        with st.spinner('Sedang menganalisis...'):
            # 1. Buat DataFrame dari input
            new_data = pd.DataFrame([input_review], columns=['Ulasan'])
            
            # 2. Terapkan pipeline preprocessing
            new_data['text_final'] = new_data['Ulasan'].astype(str).apply(normalize_text)
            new_data['text_final'] = new_data['text_final'].apply(tokenize_text)
            new_data['text_final'] = new_data['text_final'].apply(remove_stopwords)
            new_data['text_final'] = new_data['text_final'].apply(stem_text)
            new_data['text_final'] = new_data['text_final'].apply(lambda x: ' '.join(x))
            
            # 3. Ekstraksi fitur dengan TF-IDF yang sudah ada
            X_new = tfidf_vectorizer.transform(new_data['text_final'])
            
            # 4. Prediksi dengan kedua model
            pred_klasifikasi = model_klasifikasi.predict(X_new)[0]
            pred_regresi = model_regresi.predict(X_new)[0]
            
            # 5. Tampilkan hasil
            st.subheader("Hasil Analisis:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Prediksi Sentimen (Klasifikasi)")
                if pred_klasifikasi == 'positif':
                    st.success(f"**{pred_klasifikasi.upper()}** 👍")
                elif pred_klasifikasi == 'negatif':
                    st.error(f"**{pred_klasifikasi.upper()}** 👎")
                else:
                    st.warning(f"**{pred_klasifikasi.upper()}** 😐")

            with col2:
                st.markdown("##### Prediksi Skor (Regresi)")
                # Membatasi skor prediksi antara 1 dan 5
                pred_skor_final = max(1.0, min(5.0, pred_regresi))
                st.metric(label="Skor Prediksi", value=f"{pred_skor_final:.2f} / 5.0")

    elif not input_review:
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")