# diabetesprediction_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Liste des colonnes numériques (utilisée dans transform_data)
all_numerical_columns = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age",
    "Pseudo_Inverse_MI", "BMI_DiabetesPedigree", "DBP_Age_Freq_Score"
]

def remove_patient_id(dataframe):
    """Remove PatientID column from DataFrame."""
    return dataframe.drop(columns=['PatientID'])

def winsorize_with_exception(df, lower_percentile=0.05, upper_percentile=0.95):
    """
    Winsorize specified columns by replacing extreme values.
    Handles different thresholds for each column based on medical constraints.
    """
    outlier_columns = {
        'PlasmaGlucose': {'lower': True, 'upper': False},
        'DiastolicBloodPressure': {'lower': True, 'upper': False},
        'TricepsThickness': {'lower': False, 'upper': True},
        'SerumInsulin': {'lower': True, 'upper': True},
        'BMI': {'lower': False, 'upper': True}
    }

    for col, thresholds in outlier_columns.items():
        if thresholds['lower']:
            lower = df[col].quantile(lower_percentile)
            df[col] = df[col].clip(lower=lower)
        if thresholds['upper']:
            upper = df[col].quantile(upper_percentile)
            df[col] = df[col].clip(upper=upper)
    
    return df

def generate_features(df):
    """
    Generate new features including:
    - Numerical features (Pseudo_Inverse_MI, etc.)
    - Categorical features binned and one-hot encoded
    """
    # Numerical Features
    df["Pseudo_Inverse_MI"] = (df["PlasmaGlucose"] * df["SerumInsulin"]) / 10000
    df["BMI_DiabetesPedigree"] = df["BMI"] * df["DiabetesPedigree"]
    df['DBP_Age_Freq_Score'] = df['DiastolicBloodPressure'] / (df['Age'] * df['Age'].map(df['Age'].value_counts(normalize=True)))

    # Categorical Features
    df["Age_Category"] = pd.cut(df["Age"], bins=[18, 30, 60, np.inf], labels=["Young", "Adult", "Senior"])
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[18, 25, 30, np.inf], labels=["Normal", "Overweight", "Obese"])
    df["Glucose_Category"] = pd.cut(df["PlasmaGlucose"], bins=[0, 70, 130, np.inf], labels=["Hypoglycemia", "Normal", "High"])
    df["BloodPressure_Category"] = pd.cut(df["DiastolicBloodPressure"], bins=[0, 79, 89, np.inf], labels=["Normal", "Hypertension Stage 1", "Hypertension Stage 2"])
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=[
        "Age_Category", "BMI_Category", 
        "Glucose_Category", "BloodPressure_Category"
    ], drop_first=False)
    
    return df

def transform_data(df, columns=all_numerical_columns):
    """
    Apply Yeo-Johnson transformation and standardization to numerical columns.
    """
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df[columns] = pt.fit_transform(df[columns])
    return df

def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline combining all steps:
    1. Remove PatientID
    2. Handle duplicates
    3. Winsorize outliers
    4. Generate features
    5. Apply transformations
    """
    df = remove_patient_id(df)
    df = df.drop_duplicates()
    df = winsorize_with_exception(df)
    df = generate_features(df)
    df = transform_data(df)
    
    # Remove low-importance features (from original feature selection)
    features_to_remove = [
        'BloodPressure_Category_Hypertension Stage 2',
        'Glucose_Category_High',
        'BloodPressure_Category_Normal',
        'BloodPressure_Category_Hypertension Stage 1',
        'Glucose_Category_Hypoglycemia',
        'Age_Category_Senior',
        'Glucose_Category_Normal',
        'BMI_Category_Obese'
    ]
    # Only remove columns that exist
    features_to_remove = [col for col in features_to_remove if col in df.columns]
    return df.drop(columns=features_to_remove)