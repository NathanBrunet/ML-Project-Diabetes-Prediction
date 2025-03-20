import streamlit as st
import pandas as pd
import joblib  

model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
plasma_glucose = st.number_input("Plasma Glucose", min_value=0, max_value=200, step=1)
diastolic_blood_pressure = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=120, step=1)
triceps_thickness = st.number_input("Triceps Thickness", min_value=0, max_value=100, step=1)
serum_insulin = st.number_input("Serum Insulin", min_value=0, max_value=300, step=1)
bmi = st.number_input("BMI", min_value=0, max_value=50, step=1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=18, max_value=120, step=1)

if st.button("Predict"):
    input_data = [[pregnancies, plasma_glucose, diastolic_blood_pressure, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.write("The model predicts **Diabetic**.")
    else:
        st.write("The model predicts **Not Diabetic**.")
