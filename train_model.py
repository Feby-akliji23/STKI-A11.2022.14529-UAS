import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load dataset
try:
    df = pd.read_csv("./dataset/Dataset of Diabetes.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file path.")
    exit()

# Display dataset overview
print("\n### Dataset Overview ###")
print(df.head())
print("\nDataset Columns:", df.columns.tolist())

# Drop unnecessary columns
X = df.drop(['ID', 'No_Pation', 'CLASS'], axis=1, errors='ignore')
y = df['CLASS'].replace({'N': 0, 'Y': 1})  # Encode target

# Debugging kolom Gender
if 'Gender' in X.columns:
    print("\nUnique values in 'Gender' before cleaning:")
    print(X['Gender'].unique())  # Tampilkan semua nilai unik

    print("\nCleaning 'Gender' column...")
    X['Gender'] = X['Gender'].str.strip()  # Hapus spasi tambahan
    X['Gender'] = X['Gender'].str.upper()  # Ubah ke huruf kapital
    valid_values = {'F': 0, 'M': 1}
    X['Gender'] = X['Gender'].map(valid_values)

    # Tangani nilai tidak valid
    invalid_rows = X[X['Gender'].isna()]
    if not invalid_rows.empty:
        print("\nInvalid rows found in 'Gender':")
        print(invalid_rows)
        print("Dropping invalid rows...")
        X = X.drop(invalid_rows.index)

# Konversi target ke numerik
print("\nEnsuring target column 'CLASS' is numeric...")
y = pd.to_numeric(y, errors='coerce')
if y.isna().sum() > 0:
    print("\nInvalid values found in 'CLASS':")
    print(y[y.isna()])
    print("Dropping invalid target values...")
    X = X[y.notna()]
    y = y.dropna()

# Validasi tipe data
non_numeric_columns = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
if non_numeric_columns:
    print(f"\nError: Non-numeric columns found after conversion: {non_numeric_columns}.")
    exit()

# Periksa nilai yang hilang
print("\nMissing Values in Dataset:")
print(X.isna().sum())
if X.isna().sum().sum() > 0:
    print("\nFilling missing values with column means...")
    X = X.fillna(X.mean())

# Normalisasi data
print("\nNormalizing data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
print("\nSplitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
except ValueError as e:
    print("\nError during train-test split:")
    print(e)
    print("Ensure 'y' contains at least two classes.")
    exit()

# Tangani data imbalance dengan SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Save the model and scaler
dump(model, "diabetes_model.joblib")
dump(scaler, "scaler.joblib")
print("\nModel and scaler saved successfully!")
