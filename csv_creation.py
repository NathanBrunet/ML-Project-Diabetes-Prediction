import pandas as pd

# File to generate (if neededd) test data for the multiple input section of the Streamlit app

# Creating the dataset with required columns
data = {
    "Pregnancies": [2, 4],
    "PlasmaGlucose": [120, 140],
    "DiastolicBloodPressure": [80, 90],
    "TricepsThickness": [25, 32],
    "SerumInsulin": [85, 130],
    "BMI": [24.5, 29.1],
    "DiabetesPedigree": [0.5, 1.2],
    "Age": [45, 60]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Saving to CSV file
df.to_csv("test_patients.csv", index=False)
