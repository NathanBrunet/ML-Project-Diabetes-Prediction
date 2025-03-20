import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🎨 Set Streamlit Page Configuration
st.set_page_config(page_title="Diabetes Data Explorer", page_icon="📊", layout="wide")

# Custom CSS to change the background color to white and improve aesthetics
st.markdown(
    """
    <style>
    body {
        background-color: white;  /* Set to white */
        color: black;
    }
    .stButton button {
        background-color: #FFD700;  /* Gold button background */
        color: white;
    }
    .stMarkdown, .stTextInput, .stNumberInput, .stSlider, .stSelectbox, .stRadio {
        font-size: 16px;
        font-family: 'Helvetica', sans-serif;
    }
    .stSubheader, .stTitle {
        text-align: center;
    }
    .stColumns .stTextInput, .stColumns .stNumberInput {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🌟 Title with color
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎈 Diabe-Track</h1>", unsafe_allow_html=True)
st.info("Predicting Diabetes Outcomes from Health Data Patterns in the TAIPEI Dataset")

# 💂️ Sidebar for User Input
st.sidebar.header("User Input for Prediction")
age = st.sidebar.number_input("👵 Age", min_value=18, max_value=120)
pregnancies = st.sidebar.number_input("🤰 Number of Pregnancies", min_value=0)
plasma_glucose = st.sidebar.number_input("🍭 Plasma Glucose", min_value=50, max_value=250)
diastolic_bp = st.sidebar.number_input("💎 Diastolic Blood Pressure", min_value=40, max_value=200)
triceps_thickness = st.sidebar.number_input("💪 Triceps Skin Fold Thickness", min_value=10, max_value=100)
serum_insulin = st.sidebar.number_input("💉 Serum Insulin", min_value=0, max_value=1000)
bmi = st.sidebar.number_input("🏋️ BMI", min_value=10.0, max_value=60.0)
diabetes_pedigree = st.sidebar.number_input("👨‍⚕️ Diabetes Pedigree Function", min_value=0.0, max_value=2.5)

# 💂️ Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/SuzyJoelly/diabetes-predictions-app/refs/heads/main/TAIPEI_diabetes%20(1).csv")

# Remove 'PatientID' column if it exists
if 'PatientID' in df.columns:
    df = df.drop(columns=['PatientID'])

# Remove rows with missing values in the target column 'Diabetic'
df.dropna(subset=['Diabetic'], inplace=True)

# Check for missing values in the rest of the dataset
df.fillna(df.mean(), inplace=True)  # Handle missing values in the features by filling with the mean

# 🚀 **Model Building and Prediction**

# Split data into features (X) and target (y)
X = df.drop("Diabetic", axis=1)
y = df["Diabetic"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prepare the input for prediction
user_input = pd.DataFrame([[pregnancies, plasma_glucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age]],
                          columns=["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", 
                                   "SerumInsulin", "BMI", "DiabetesPedigree", "Age"])

# Ensure the input has the same feature names and order as the training data
user_input = user_input[X.columns.tolist()]  # Make sure columns are ordered as in X

# Standardize the user input based on the scaler fitted to the training data
user_input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(user_input_scaled)

# 📊 **Data Visualization Section**
st.subheader("📊 Data Visualizations")

# 🔹 Custom Color Palette
colors = ["#FF4B4B", "#1E88E5", "#FFC107", "#2E7D32", "#D81B60", "#8E24AA"]

# **1. Diabetes Count Plot (Improved Aesthetic)**
st.write("###  Diabetes Cases")
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=df["Diabetic"], palette=["#1E88E5", "#D81B60"], ax=ax)
ax.set_xticklabels(["No Diabetes", "Diabetes"])
ax.set_ylabel("Count")
ax.set_xlabel("Diabetic Status")
ax.set_title("Diabetes Cases", fontsize=14)
st.pyplot(fig)

st.write("### 🎂 Age Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True, color="#FF4B4B", edgecolor="black")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

st.write("### 🍬 Plasma Glucose/Age")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df["Age"], y=df["PlasmaGlucose"], hue=df["Diabetic"], palette=["#1E88E5", "#D81B60"], alpha=0.7)
ax.set_xlabel("Age")
ax.set_ylabel("Plasma Glucose Level")
st.pyplot(fig)

# Move prediction to sidebar
with st.sidebar.expander("Prediction Result"):
    if prediction == 1:
        st.markdown(f"<h3 style='color: #D81B60; text-align: center;'>🚨 Prediction: The patient is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: #1E88E5; text-align: center;'>✅ Prediction: The patient is likely to not have diabetes.</h3>", unsafe_allow_html=True)
