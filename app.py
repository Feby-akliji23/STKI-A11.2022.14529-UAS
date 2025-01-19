import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# User-defined Functions
def val_count(data: pd.Series, orderby: str = 'Count', ascending: bool = False) -> pd.DataFrame:
    if type(data) == pd.Series:
        result = data.value_counts(dropna=False).reset_index()
        result.columns = ['Values', 'Count']
        result['%'] = np.around(result['Count'] * 100 / len(data), 3)
        return result.sort_values(by=orderby, ascending=ascending)
    else:
        return 'Input Series only'

def numerical_summary(data: pd.Series, n: int = 3) -> pd.Series:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    range_value = np.abs(max_value - min_value)
    skewness = data.skew()
    kurtosis = data.kurtosis()
    result = pd.Series([min_value, max_value, range_value, mean, median, std, skewness, kurtosis])
    result.index = ['min', 'max', 'range', 'mean', 'median', 'std', 'skewness', 'kurtosis']
    return np.around(result, n)

# Streamlit App
st.title("Diabetes Prediction Dashboard")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.dataframe(data)

    # Data Cleaning
    st.write("### Data Cleaning")
    st.write("#### Removing Unnecessary Data")
    st.write(f"Columns Before ({len(data.columns)}):", data.columns.tolist())
    data.drop(['ID', 'No_Pation'], axis=1, inplace=True)
    st.write(f"Columns After ({len(data.columns)}):", data.columns.tolist())

    st.write("#### Correcting Data Entry Errors")
    data['Gender'] = data['Gender'].replace({'f': 'F'})
    data['CLASS'] = data['CLASS'].replace({'N ': 'N', 'P': 'Y', 'Y ': 'Y'})
    st.write(data.select_dtypes('object').nunique())

    # Handling Missing Values
    st.write("### Handling Missing Values")
    st.write(data.isna().sum())

    # Dealing with Duplicates
    st.write("### Removing Duplicate Records")
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    after = len(data)
    st.write(f"Records Before: {before}, Records After: {after}")

    # Data Transformation
    st.write("### Data Transformation")
    st.write("#### Encode Gender and CLASS")
    data['Gender'].replace({'F': 0, 'M': 1}, inplace=True)
    data['CLASS'].replace({'N': 0, 'Y': 1}, inplace=True)
    st.write(data.head())

    # Data Exploration
    st.write("### Data Exploration")
    st.write("#### Correlation Heatmap")
    plt.figure(figsize=(11, 8))
    sns.heatmap(data.corr(method='spearman'), annot=True, cmap='Blues')
    st.pyplot(plt)

    st.write("#### Univariate Analysis (AGE)")
    sns.histplot(data['AGE'], kde=True, color='dodgerblue')
    plt.title('AGE Distribution')
    st.pyplot(plt)

    # Feature Engineering
    st.write("### Feature Engineering")
    norm_col = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[norm_col])
    data_scaled = pd.DataFrame(data_scaled, columns=norm_col)
    data = pd.concat([data_scaled, data[['Gender', 'CLASS']]], axis=1)
    st.write("Normalized Data")
    st.dataframe(data)

    # Train-Test Split
    st.write("### Train-Test Split")
    X = data.drop('CLASS', axis=1)
    y = data['CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    st.write(f"Training Data: {len(X_train)}, Testing Data: {len(X_test)}")

    # Resampling Data
    st.write("### Resampling Data with SMOTE")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_SMOTE, y_SMOTE = smote.fit_resample(X_train, y_train)
    st.write("After SMOTE:")
    st.write(val_count(pd.Series(y_SMOTE)))

    # Modeling
    st.write("### Model Training and Evaluation")
    resample = [
        ('Original', X_train, y_train),
        ('SMOTE', X_SMOTE, y_SMOTE)
    ]
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('XGBoost', XGBClassifier(random_state=42))
    ]

    results = []
    for r_name, X_res, y_res in resample:
        for m_name, model in models:
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            results.append((r_name + ' - ' + m_name, accuracy, precision, recall, f1))

    comparison = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    st.write("Model Evaluation")
    st.dataframe(comparison.sort_values(by='F1 Score', ascending=False))

