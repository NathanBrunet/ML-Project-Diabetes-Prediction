# diabetesprediction_utils.py
import pandas as pd
import numpy as np

all_numerical_columns = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
    "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age",
    "Pseudo_Inverse_MI", "BMI_DiabetesPedigree", "DBP_Age_Freq_Score"
]

def remove_patient_id(dataframe):
    """Remove PatientID column if it exists"""
    if 'PatientID' in dataframe.columns:
        return dataframe.drop(columns=['PatientID'])
    return dataframe

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

def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline combining all steps
    """
    df = remove_patient_id(df)
    df = generate_features(df)
    
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

"""test_data = pd.DataFrame({
    "Pregnancies": [2, 4],
    "PlasmaGlucose": [120, 140],
    "DiastolicBloodPressure": [80, 90],
    "TricepsThickness": [25, 32],
    "SerumInsulin": [85, 130],
    "BMI": [24.5, 29.1],
    "DiabetesPedigree": [0.5, 1.2],
    "Age": [45, 60]
})

print("Dataset before preprocessing:")
print(test_data)

processed_data = preprocess_pipeline(test_data)

print("\nDataset after preprocessing:")
print(processed_data)

new_columns = [col for col in processed_data.columns if col not in test_data.columns]

print("\nList of columns in the processed dataset:")
print(processed_data.columns.tolist())"""