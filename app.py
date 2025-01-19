import streamlit as st
import numpy as np
from joblib import load

# Load pre-trained model dan scaler
try:
    model = load("diabetes_model.joblib")
    scaler = load("scaler.joblib")
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please train the model and save it first.")
    st.stop()

# Judul aplikasi
st.title("Diabetes Prediction Dashboard")

st.markdown("""
Masukkan nilai untuk fitur-fitur berikut untuk memprediksi apakah seseorang terindikasi **Diabetes** atau **Non-Diabetes**.
Contoh nilai:
- **Pregnancies**: 6+ untuk diabetes, 0-2 untuk non-diabetes
- **Glucose**: >140 untuk diabetes, <100 untuk non-diabetes
- **Blood Pressure**: >90 untuk diabetes, <80 untuk non-diabetes
- **BMI**: >30 untuk diabetes, 18-25 untuk non-diabetes
""")

# Input untuk prediksi
pregnancies = st.number_input("Enter the Pregnancies value", value=0.0, step=1.0)
glucose = st.number_input("Enter the Glucose value", value=0.0, step=1.0)
blood_pressure = st.number_input("Enter the Blood Pressure value", value=0.0, step=1.0)
skin_thickness = st.number_input("Enter the Skin Thickness value", value=0.0, step=1.0)
insulin = st.number_input("Enter the Insulin value", value=0.0, step=1.0)
bmi = st.number_input("Enter the BMI value", value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Enter the Diabetes Pedigree Function value", value=0.0, step=0.01)
age = st.number_input("Enter the Age value", value=0.0, step=1.0)

# Menambahkan fitur dummy untuk memenuhi input scaler
# Asumsi bahwa scaler dilatih dengan fitur tambahan seperti Gender (0 untuk default)
dummy_features = [0.0, 0.0, 0.0]  # Sesuaikan jumlah dummy sesuai dengan fitur yang dihapus
input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age] + dummy_features

# Prediksi
if st.button("Predict Diabetes"):
    # Normalisasi input data
    try:
        input_data = scaler.transform([input_features])  # Pastikan input_data sesuai dengan scaler
        prediction = model.predict(input_data)
        result = "Diabetes" if prediction[0] == 1 else "Non-Diabetes"
        st.success(f"The prediction result is: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
