# %% [markdown]
# ## Initialization

# %% [markdown]
# ### Importing Libraries

# %%

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns


# Feature Engineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Imbalance Dataset
from imblearn.over_sampling import SMOTE

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report #, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %% [markdown]
# ### User-defined Functions

# %%
def val_count(data:pd.Series, orderby:str='Count', ascending:bool=False) -> pd.DataFrame:
    '''
    Return a DataFrame containing counts of unique rows in Series

    Parameters
    ----------
    data: Series
        the data to be displayed value counts
    orderby: str (Count or Values)
        how to order the data, default Count
    ascending: bool (default: False)
        True for ascending, False for descending

    Returns
    ------
    DataFrame
        - Values : Values name
        - Count : Count per value
        - % : Percentage of count
    '''
    if type(data) == pd.Series:
        result = data.value_counts(dropna=False).reset_index() # without drop
        result.columns = ['Values', 'Count']
        result['%'] = np.around(result['Count'] * 100/len(data), 3)
        return result.sort_values(by=orderby, ascending=ascending)
    else:
        return 'Input Series only'

# %%
def numerical_summary(data: pd.Series, n: int = 3) -> pd.Series:
    '''
    Statistics summary for numerical data.

    Parameters
    ----------
    data : pandas series
        the data to be displayed in summary
    n : int, optional
        determine the number after the comma of the result
    '''

    # central tendency: mean, median
    mean = np.mean(data)
    median = np.median(data)

    # # distribution: std, min, max, range, skew, kurtosis
    std = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    range_value = np.abs(max_value - min_value)
    skewness = data.skew()
    kurtosis = data.kurtosis()

    # # concatenates
    result = pd.Series([min_value, max_value, range_value, mean, median, std, skewness, kurtosis])
    result.index = ['min','max', 'range','mean','median',  'std','skewness','kurtosis']

    return np.around(result, n)
    # return mean

# %% [markdown]
# ## Data Collecting

# %%
path = './dataset/Dataset of Diabetes.csv'
data = pd.read_csv(path)
data

# %% [markdown]
# ## Data Cleaning and Preprocessing

# %% [markdown]
# ### Removing Unnecessary Data

# %%
print(f'Columns Before ({len(data.columns)}):', data.columns.tolist())
data.drop(['ID','No_Pation'], axis=1, inplace=True)
print(f'Columns After ({len(data.columns)}):', data.columns.tolist())

# %% [markdown]
# ### Correcting Data Entry Errors

# %%
print('# Before')
for col in data.select_dtypes('object').columns:
    print(f'{col}: {data[col].unique().tolist()}')

# %%
data['Gender'] =  data['Gender'].replace({'f': 'F'})
data['CLASS'] =  data['CLASS'].replace({'N ': 'N', 'P': 'Y', 'Y ': 'Y'})

# %%
print('# After')
for col in data.select_dtypes('object').columns:
    print(f'{col}: {data[col].unique().tolist()}')

# %% [markdown]
# ### Handling Missing Value

# %%
data.isna().sum()

# %% [markdown]
# ### Dealing with Duplicate Records

# %%
print('Records Before:', len(data))
data = data.drop_duplicates().reset_index(drop=True)
print('Records After:', len(data))

# %% [markdown]
# ### Data Transformation

# %% [markdown]
# #### Encode Gender

# %%
print('Before:', data['Gender'].unique().tolist())
data['Gender'].replace({'F': 0, 'M': 1}, inplace=True)
print('After:', data['Gender'].unique().tolist())

# %% [markdown]
# #### Encode CLASS

# %%
print('Before:', data['CLASS'].unique().tolist())
data['CLASS'].replace({'N': 0, 'Y': 1}, inplace=True)
print('After:', data['CLASS'].unique().tolist())

# %% [markdown]
# ## Data Exploration

# %% [markdown]
# ### Univariate Analysis (Categorical)

# %% [markdown]
# #### Gender

# %%
gender = val_count(data.Gender)
gender.style.hide(axis='index')

# %%
ax = sns.barplot(data=gender, x='Values', y='Count', color='dodgerblue')
ax.bar_label(ax.containers[0], fmt='%i', padding=-25, color='white', fontweight='bold')
plt.title('Gender', fontweight='bold')
plt.show()

# %% [markdown]
# ### Univariate Analysis (Numerical)

# %% [markdown]
# #### AGE

# %%
sns.histplot(data['AGE'], kde=True, color='dodgerblue')
plt.title('AGE', fontweight='bold')
plt.show()

# %%
numerical_summary(data['AGE'])

# %% [markdown]
# #### Urea

# %%
sns.histplot(data['Urea'], kde=True, color='dodgerblue')
plt.title('Urea', fontweight='bold')
plt.show()

# %%
numerical_summary(data['Urea'])

# %% [markdown]
# #### Cr

# %%
sns.histplot(data['Cr'], kde=True, color='dodgerblue')
plt.title('Cr', fontweight='bold')
plt.show()

# %%
numerical_summary(data['Cr'])

# %% [markdown]
# #### HbA1c

# %%
sns.histplot(data['HbA1c'], kde=True, color='dodgerblue')
plt.title('HbA1c', fontweight='bold')
plt.show()

# %%
numerical_summary(data['HbA1c'])

# %% [markdown]
# #### Chol

# %%
sns.histplot(data['Chol'], kde=True, color='dodgerblue')
plt.title('Chol', fontweight='bold')
plt.show()

# %%
numerical_summary(data['Chol'])

# %% [markdown]
# #### TG

# %%
sns.histplot(data['TG'], kde=True, color='dodgerblue')
plt.title('TG', fontweight='bold')
plt.show()

# %%
numerical_summary(data['TG'])

# %% [markdown]
# #### HDL

# %%
sns.histplot(data['HDL'], kde=True, color='dodgerblue')
plt.title('HDL', fontweight='bold')
plt.show()

# %%
numerical_summary(data['HDL'])

# %% [markdown]
# #### LDL

# %%
sns.histplot(data['LDL'], kde=True, color='dodgerblue')
plt.title('LDL', fontweight='bold')
plt.show()

# %%
numerical_summary(data['LDL'])

# %% [markdown]
# #### VLDL

# %%
sns.histplot(data['VLDL'], kde=True, color='dodgerblue')
plt.title('VLDL', fontweight='bold')
plt.show()

# %%
numerical_summary(data['VLDL'])

# %% [markdown]
# #### BMI

# %%
sns.histplot(data['BMI'], kde=True, color='dodgerblue')
plt.title('BMI', fontweight='bold')
plt.show()

# %%
numerical_summary(data['BMI'])

# %% [markdown]
# ### Correlation

# %%
plt.figure(figsize=(11,8))
sns.heatmap(data.corr(method='spearman'), annot=True, cmap='Blues')
plt.show()

# %%
corr_with_target = data.drop('CLASS', axis=1).corrwith(data['CLASS'],axis=0, method='spearman')
corr_with_target = corr_with_target.sort_values(key=abs, ascending=False).reset_index()
corr_with_target = corr_with_target.rename(columns={'index':'Feature', 0:'Corr'})
print('Sorted Correlation Between Feature and Target(CLASS)\n')
display(corr_with_target.style.hide(axis='index'))

# %% [markdown]
# ### Label Distributions

# %%
class_ = val_count(data.CLASS)
class_.style.hide(axis='index')

# %%
ax = sns.barplot(data=class_, x='Values', y='Count', color='dodgerblue')
ax.bar_label(ax.containers[0], fmt='%i', padding=-25, color='white', fontweight='bold')
plt.title('CLASS', fontweight='bold')
plt.show()

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Normalizing

# %%
norm_col = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[norm_col])
data_scaled = pd.DataFrame(data_scaled, columns=norm_col)
data = pd.concat([data_scaled, data[['Gender', 'CLASS']]], axis=1)
data

# %% [markdown]
# ### Train-Test Split

# %%
X = data.drop('CLASS', axis=1)
y = data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42)

# X_train.shape, X_test.shape

print('Jumlah Baris Data Latih:', len(X_train))
print('Jumlah Baris Data Uji:', len(X_test))

# %% [markdown]
# ### Resampling Data for Imbalance Dataset

# %%
# Initialize & fit resample
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_SMOTE, y_SMOTE = smote.fit_resample(X_train, y_train)

# Counting labels
sampler_count = [['Original Data', y_train], ['SMOTE', y_SMOTE]]

# Print results
for i in sampler_count:
    print(i[0])
    display(val_count(pd.Series(i[1])))

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Model Fit & Predict

# %%
# Initialize resampling data
resample = [
    ('Original', X_train, y_train),
    ('SMOTE', X_SMOTE, y_SMOTE)
]

# Initialize models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42))
]

# Evaluate models
results = []

# Fit & Predict
for r_name, X, y in resample:
    for m_name, model in models:
        model.fit(X, y)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((r_name+' - '+m_name, accuracy, precision, recall, f1))

# %% [markdown]
# ## Evaluation

# %%
# Create comparison table
comparison = pd.DataFrame(results, columns=['Model', 'Accuracy','Precision',
                                            'Recall', 'F1 Score'])
comparison = comparison.set_index(['Model'])
comparison.index.name = None

# Print sorted comparison table
comparison.reindex(comparison.mean(axis=1).sort_values(ascending=False).index)


