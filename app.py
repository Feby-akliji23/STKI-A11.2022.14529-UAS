import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Fungsi untuk melatih model
def train_models(X_train, y_train, X_test, y_test):
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('XGBoost', XGBClassifier(random_state=42))
    ]
    results = []
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((model_name, accuracy, precision, recall, f1))
    return results, models[-1][1]  # Mengembalikan hasil dan model terbaik (XGBoost)

# Fungsi prediksi
def predict_diabetes(model, input_data):
    prediction = model.predict(input_data)
    return "Diabetes" if prediction[0] == 1 else "Non-Diabetes"

# Judul aplikasi
st.title("Diabetes Prediction Dashboard")

# Memuat dataset
try:
    st.write("### Loading Dataset...")
    df = pd.read_csv("./dataset/Dataset of Diabetes.csv")  # Pastikan path benar
    st.write("### Dataset Overview")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("Dataset not found. Please check the file path.")
    st.stop()

# Preprocessing
try:
    st.write("### Preprocessing Data")
    # Drop kolom yang tidak relevan
    X = df.drop(['ID', 'No_Pation', 'CLASS'], axis=1, errors='ignore')
    y = df['CLASS'].replace({'N': 0, 'Y': 1})  # Encode target

    # Periksa missing values
    missing_values = X.isna().sum()
    st.write("Missing Values in Columns:")
    st.write(missing_values)

    # Isi missing values jika ada
    if missing_values.sum() > 0:
        st.warning("Missing values detected. Filling with column mean...")
        X = X.fillna(X.mean())

    # Periksa tipe data
    non_numeric_columns = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
    if non_numeric_columns:
        st.error(f"Non-numeric columns detected: {non_numeric_columns}. Please clean your dataset.")
        st.stop()

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("Data normalized successfully.")
except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# Train-test split
st.write("### Splitting Data into Train and Test Sets")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Resampling dengan SMOTE
st.write("### Handling Imbalanced Dataset with SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
st.success(f"Data after SMOTE: {len(X_train_smote)} samples for training.")

# Latih model dan evaluasi
st.write("### Training Models and Evaluating Performance")
results, best_model = train_models(X_train_smote, y_train_smote, X_test, y_test)
comparison = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
st.write("### Model Evaluation Results")
st.dataframe(comparison.sort_values(by='F1 Score', ascending=False))

# Input untuk prediksi
st.write("### Predict Diabetes")
pregnancies = st.number_input("Enter the Pregnancies value (e.g., 6+ for diabetes, 0-2 for non-diabetes)", value=0.0, step=1.0)
glucose = st.number_input("Enter the Glucose value (e.g., > 140 for diabetes, < 100 for non-diabetes)", value=0.0, step=1.0)
blood_pressure = st.number_input("Enter the Blood Pressure value (e.g., > 90 for diabetes, < 80 for non-diabetes)", value=0.0, step=1.0)
skin_thickness = st.number_input("Enter the Skin Thickness value (e.g., > 30 for diabetes, < 20 for non-diabetes)", value=0.0, step=1.0)
insulin = st.number_input("Enter the Insulin value (e.g., > 200 for diabetes, < 100 for non-diabetes)", value=0.0, step=1.0)
bmi = st.number_input("Enter the BMI value (e.g., > 30 for diabetes, 18-25 for non-diabetes)", value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Enter the Diabetes Pedigree Function value (e.g., > 0.8 for diabetes, < 0.5 for non-diabetes)", value=0.0, step=0.01)
age = st.number_input("Enter the Age value (e.g., > 50 for diabetes, < 30 for non-diabetes)", value=0.0, step=1.0)

if st.button("Diabetes Prediction Test"):
    try:
        # Normalisasi input data
        input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        # Prediksi menggunakan model terbaik
        result = predict_diabetes(best_model, input_data)
        st.success(f"The prediction result is: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
