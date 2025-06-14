# 🧠 Capstone Project: Prediksi Kategori Obesitas

Proyek ini merupakan aplikasi prediksi kategori obesitas berbasis **Machine Learning** menggunakan model **Random Forest Classifier**, serta antarmuka visual interaktif dengan **Streamlit**. Model dilatih menggunakan dataset obesitas dari Meksiko, Peru, dan Kolombia.

---

## 📌 Tujuan Proyek

Memprediksi **kategori obesitas** seseorang berdasarkan data gaya hidup, kebiasaan makan, aktivitas fisik, serta faktor genetik.

---

## 📊 Dataset

Dataset: **ObesityDataSet.csv**

- Jumlah data: 2111 baris
- Fitur: 16 fitur input, 1 target (`NObeyesdad`)
- Target (`NObeyesdad`) terdiri dari 7 kelas:
  - Insufficient_Weight
  - Normal_Weight
  - Overweight_Level_I
  - Overweight_Level_II
  - Obesity_Type_I
  - Obesity_Type_II
  - Obesity_Type_III

Contoh fitur:
- `Age`, `Gender`, `Height`, `Weight`
- `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`
- `FAF`, `TUE`, `CALC`, `MTRANS`
- `family_history_with_overweight`

---

## ⚙️ Machine Learning

- **Model**: Random Forest Classifier  
- **Akurasi**: ~90% setelah preprocessing dan tuning  
- Teknik:
  - SMOTE oversampling
  - Label encoding + One Hot Encoding
  - Outlier & duplicate removal
  - Hyperparameter tuning (`GridSearchCV`)

Model disimpan sebagai file:
- `rf_model.sav`
- `scaler.pkl`
