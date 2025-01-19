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
""")

# Input untuk prediksi
def input_with_example(label, default, step, help_text):
    return st.number_input(label, value=default, step=step, help=help_text)

pregnancies = input_with_example(
    "Pregnancies",
    default=0.0,
    step=1.0,
    help_text="Contoh: 6+ untuk kemungkinan diabetes, 0-2 untuk non-diabetes"
)

glucose = input_with_example(
    "Glucose",
    default=0.0,
    step=1.0,
    help_text="Contoh: >140 untuk kemungkinan diabetes, <100 untuk non-diabetes"
)

blood_pressure = input_with_example(
    "Blood Pressure",
    default=0.0,
    step=1.0,
    help_text="Contoh: >90 untuk kemungkinan diabetes, <80 untuk non-diabetes"
)

skin_thickness = input_with_example(
    "Skin Thickness",
    default=0.0,
    step=1.0,
    help_text="Contoh: >32 untuk kemungkinan diabetes, <20 untuk non-diabetes"
)

insulin = input_with_example(
    "Insulin",
    default=0.0,
    step=1.0,
    help_text="Contoh: >200 untuk kemungkinan diabetes, <100 untuk non-diabetes"
)

bmi = input_with_example(
    "BMI",
    default=0.0,
    step=0.1,
    help_text="Contoh: >30 untuk kemungkinan diabetes, 18-25 untuk non-diabetes"
)

diabetes_pedigree_function = input_with_example(
    "Diabetes Pedigree Function",
    default=0.0,
    step=0.01,
    help_text="Contoh: >0.5 untuk kemungkinan diabetes, <0.3 untuk non-diabetes"
)

age = input_with_example(
    "Age",
    default=0.0,
    step=1.0,
    help_text="Contoh: >50 untuk kemungkinan diabetes, <30 untuk non-diabetes"
)

# Placeholder for unused features
# Jika model Anda dilatih dengan lebih banyak fitur, tambahkan nilai default untuk fitur yang hilang
unused_features = [0.0, 0.0, 0.0]  # Sesuaikan jumlah sesuai dengan kebutuhan

# Prediksi
if st.button("Predict Diabetes"):
    # Normalisasi input data
    try:
        input_data = scaler.transform([
            [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age] + unused_features
        ])
        prediction = model.predict(input_data)
        result = "Diabetes" if prediction[0] == 1 else "Non-Diabetes"
        st.success(f"The prediction result is: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
