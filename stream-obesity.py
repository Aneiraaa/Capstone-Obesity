import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Judul Web
st.title('Capstone Project Obesity')

# Load model dan preprocessing tools
rf_model = pickle.load(open('rf_model.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model_features = pickle.load(open('feature_columns.pkl', 'rb'))
numerical_cols = pickle.load(open('numerical_columns.pkl', 'rb'))

st.subheader("Masukkan Data Individu")

# Form input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Umur", min_value=1, max_value=100, value=25)
height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.65)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=65.0)

favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["yes", "no"])
fcvc = st.slider("Frekuensi konsumsi sayur per hari", 1, 3, 2)
ncp = st.slider("Jumlah makanan utama per hari", 1, 4, 3)
caec = st.selectbox("Kebiasaan ngemil", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.slider("Konsumsi air putih (liter per hari)", 1, 3, 2)
scc = st.selectbox("Mengonsumsi produk diet?", ["yes", "no"])
faf = st.slider("Frekuensi aktivitas fisik mingguan (jam)", 0, 7, 2)
tue = st.slider("Waktu layar (jam per hari)", 0, 5, 1)
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Bike", "Motorbike"])
fhwo = st.selectbox("Riwayat keluarga overweight?", ["yes", "no"])

# Jika tombol ditekan
if st.button("Prediksi Kategori Obesitas"):
    # Susun input ke DataFrame
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'CALC': [calc],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'SCC': [scc],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'family_history_with_overweight': [fhwo],
        'FAF': [faf],
        'TUE': [tue],
        'CAEC': [caec],
        'MTRANS': [mtrans]
    })

    # Pisahkan numerik dan kategorikal
    input_num = input_df[numerical_cols].copy()
    input_cat = input_df.drop(columns=numerical_cols)

    # Scaling numerik
    input_num_scaled = pd.DataFrame(scaler.transform(input_num), columns=numerical_cols)

    # One-hot encoding kategorikal
    input_cat_encoded = pd.get_dummies(input_cat)

    # Gabung semua fitur
    input_combined = pd.concat([input_num_scaled, input_cat_encoded], axis=1)

    # Tambahkan fitur kosong untuk yang tidak muncul
    for col in model_features:
        if col not in input_combined.columns:
            input_combined[col] = 0

    # Pastikan urutan kolom sama
    input_combined = input_combined[model_features]

    # Prediksi
    prediction = rf_model.predict(input_combined)[0]
    st.success(f"Kategori Obesitas yang Diprediksi: {prediction.replace('_', ' ').title()}")
