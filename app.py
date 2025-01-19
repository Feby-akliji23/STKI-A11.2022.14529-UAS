import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi prediksi
def predict_diabetes(model, input_data):
    prediction = model.predict(input_data)
    return "Diabetes" if prediction[0] == 1 else "Non-Diabetes"

# Judul aplikasi
st.title("Diabetes Prediction")

st.markdown("**Masukkan nilai untuk masing-masing fitur.**")
st.markdown("Contoh nilai: **Nilai tinggi (positif diabetes)** dan **nilai rendah (non-diabetes)** diberikan untuk panduan.")

# Input untuk fitur dengan placeholder
pregnancies = st.number_input(
    "Enter the Pregnancies value (e.g., 6+ for diabetes, 0-2 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)
glucose = st.number_input(
    "Enter the Glucose value (e.g., > 140 for diabetes, < 100 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)
blood_pressure = st.number_input(
    "Enter the Blood Pressure value (e.g., > 90 for diabetes, < 80 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)
skin_thickness = st.number_input(
    "Enter the Skin Thickness value (e.g., > 30 for diabetes, < 20 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)
insulin = st.number_input(
    "Enter the Insulin value (e.g., > 200 for diabetes, < 100 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)
bmi = st.number_input(
    "Enter the BMI value (e.g., > 30 for diabetes, 18-25 for non-diabetes)", 
    min_value=0.0, 
    step=0.1,
    value=0.0
)
diabetes_pedigree_function = st.number_input(
    "Enter the Diabetes Pedigree Function value (e.g., > 0.8 for diabetes, < 0.5 for non-diabetes)", 
    min_value=0.0, 
    step=0.01,
    value=0.0
)
age = st.number_input(
    "Enter the Age value (e.g., > 50 for diabetes, < 30 for non-diabetes)", 
    min_value=0.0, 
    step=1.0,
    value=0.0
)

# Tombol prediksi
if st.button("Diabetes Prediction Test"):
    # Dataset dummy untuk pelatihan model
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 10, size=100),
        'Glucose': np.random.uniform(50, 200, size=100),
        'BloodPressure': np.random.uniform(50, 150, size=100),
        'SkinThickness': np.random.uniform(10, 50, size=100),
        'Insulin': np.random.uniform(15, 276, size=100),
        'BMI': np.random.uniform(18, 50, size=100),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, size=100),
        'Age': np.random.randint(20, 80, size=100),
        'Outcome': np.random.choice([0, 1], size=100)
    })

    # Pisahkan fitur dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # SMOTE untuk data imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Model Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)

    # Input data untuk prediksi
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Prediksi hasil
    result = predict_diabetes(model, input_data)
    st.success(f"The prediction result is: {result}")
