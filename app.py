import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Kebahagiaan Negara", layout="centered")

st.title("🌍 Prediksi Tingkat Kebahagiaan Negara")
st.write("Aplikasi ini menggunakan model Regresi Linear Berganda untuk memprediksi apakah suatu negara tergolong **Bahagia (Ya)** atau **Tidak Bahagia (Tidak)** berdasarkan indikator ekonomi dan sosial.")

# 2. Fungsi untuk memuat model (menggunakan cache agar tidak dimuat berulang kali)
@st.cache_resource
def load_models():
    with open('fitur.pkl', 'rb') as f:
        fitur = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('regresi_berganda.pkl', 'rb') as f:
        model = pickle.load(f)
    return fitur, scaler, model

# Memuat file pickle
try:
    fitur, scaler, model = load_models()
except Exception as e:
    st.error("Gagal memuat model! Pastikan file 'fitur.pkl', 'scaler.pkl', dan 'regresi_berganda.pkl' berada di folder yang sama dengan app.py.")
    st.stop()

# 3. Membuat Form Input untuk User
st.subheader("📊 Masukkan Data Prediktor")

input_data = {}
# Membagi tampilan input menjadi 2 kolom agar lebih rapi
col1, col2 = st.columns(2)

# Looping berdasarkan daftar fitur yang ada di fitur.pkl
for i, col_name in enumerate(fitur):
    if i % 2 == 0:
        with col1:
            # Gunakan st.number_input untuk menerima input angka desimal
            input_data[col_name] = st.number_input(f"Nilai {col_name}", value=0.00, format="%.4f")
    else:
        with col2:
            input_data[col_name] = st.number_input(f"Nilai {col_name}", value=0.00, format="%.4f")

# 4. Tombol Prediksi
st.markdown("---")
if st.button("🚀 Prediksi Sekarang", use_container_width=True):
    
    # a. Ubah input dari dictionary ke dalam format DataFrame (sesuai format data training)
    df_input = pd.DataFrame([input_data])
    
    # b. Standarisasi/Scaling data menggunakan scaler.pkl yang sudah dilatih
    scaled_input = scaler.transform(df_input)
    
    # c. Melakukan prediksi menggunakan model regresi
    prediksi_kontinu = model.predict(scaled_input)[0]
    
    # d. Mengonversi hasil regresi (angka desimal) menjadi label klasifikasi dengan batas 0.5
    if prediksi_kontinu >= 0.5:
        status = "BAHAGIA (Ya)"
        st.success(f"### Hasil: Negara ini diprediksi **{status}**")
        st.balloons() # Animasi balon untuk hasil positif
    else:
        status = "TIDAK BAHAGIA (Tidak)"
        st.error(f"### Hasil: Negara ini diprediksi **{status}**")
        
    # Menampilkan nilai desimal asli sebagai informasi tambahan
    st.info(f"Nilai perhitungan regresi mentah: **{prediksi_kontinu:.4f}**")
