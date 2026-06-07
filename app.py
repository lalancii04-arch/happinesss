import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Happiness Score Predictor", layout="wide")

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file excel berada di folder yang sama dengan app.py
    df = pd.read_excel('The Economics of Happiness.xlsx')
    
    # Preprocessing berdasarkan notebook
    kolom_dibuang = ['Happiness rank', 'Label_Angka', 'Year']
    df = df.drop(columns=kolom_dibuang, errors='ignore')
    
    # Menghapus missing values yang terdeteksi di notebook
    df = df.dropna()
    return df

df = load_data()

# --- MODEL TRAINING (Linear Regression) ---
# Menggunakan fitur yang tersisa untuk memprediksi Happiness Score
TARGET = 'Happiness Score'
X = df.drop(columns=[TARGET, 'Country'], errors='ignore')
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["Eksplorasi Data (EDA)", "Prediksi Happiness Score"])

# --- MENU 1: EDA ---
if menu == "Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data (EDA)")
    st.write("Aplikasi ini menggunakan dataset **The Economics of Happiness**.")
    
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Distribusi {TARGET}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[TARGET], kde=True, color='steelblue', ax=ax)
        ax.set_title(f'Distribusi {TARGET}')
        ax.set_xlabel(TARGET)
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Heatmap Korelasi")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax2)
        ax2.set_title('Heatmap Korelasi Pearson')
        st.pyplot(fig2)

# --- MENU 2: PREDIKSI ---
elif menu == "Prediksi Happiness Score":
    st.title("🔮 Prediksi Happiness Score")
    st.write("Masukkan nilai metrik negara pada form di bawah ini untuk memprediksi **Happiness Score**.")
    
    # Membuat form input berdasarkan nilai min & max dari dataset
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gdp = st.slider("GDP per capita", float(X['GDP per capita'].min()), float(X['GDP per capita'].max()), float(X['GDP per capita'].mean()))
            social = st.slider("Social support", float(X['Social support'].min()), float(X['Social support'].max()), float(X['Social support'].mean()))
            healthy = st.slider("Healthy life", float(X['Healthy life'].min()), float(X['Healthy life'].max()), float(X['Healthy life'].mean()))
            
        with col2:
            freedom = st.slider("Freedom", float(X['Freedom'].min()), float(X['Freedom'].max()), float(X['Freedom'].mean()))
            generosity = st.slider("Generosity", float(X['Generosity'].min()), float(X['Generosity'].max()), float(X['Generosity'].mean()))
            corruption = st.slider("Corruption", float(X['Corruption'].min()), float(X['Corruption'].max()), float(X['Corruption'].mean()))
            
        submitted = st.form_submit_button("Prediksi!")
        
        if submitted:
            # Memasukkan input user ke dalam format DataFrame
            user_data = pd.DataFrame({
                'GDP per capita': [gdp],
                'Social support': [social],
                'Healthy life': [healthy],
                'Freedom': [freedom],
                'Generosity': [generosity],
                'Corruption': [corruption]
            })
            
            # Melakukan prediksi
            prediction = model.predict(user_data)[0]
            
            st.success(f"### Estimasi Happiness Score: **{prediction:.3f}**")