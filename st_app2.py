import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from diabetesprediction_utils import preprocess_pipeline

st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🏥")

# Load trained model
@st.cache_data
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# Custom Theme
custom_css = """
<style>
    body { background-color: #FFFFFF; color: #333333; }
    .stApp { background-color: #FFFFFF; }
    .css-18e3th9 { background-color: #F5F5F5; }  /* Sidebar */
    .st-bb { color: #4BC9FF !important; }  /* Primary Color */
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------- Sidebar UI  ----------------
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.image("logo.jpg", width=120)
st.sidebar.markdown("<h2 style='text-align: center;'>Diabetes Prediction</h2>", unsafe_allow_html=True)

# Sidebar Menu selectbox
menu = st.sidebar.selectbox("📌 Select a Page", 
                          ["🏥 Prediction", "📊 Study Report", "📈 Model Evaluation", "ℹ️ About"])

# ----------------------  PREDICTION ----------------------
if menu == "🏥 Prediction":
    st.title("🩺 Diabetes Prediction")
    prediction_type = st.radio("Choose Prediction Type:", 
                             ["Single Prediction", "Multiple Prediction"], 
                             horizontal=True)

    if prediction_type == "Single Prediction":
        st.subheader("📝 Enter Patient Details")
        
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.slider("🤰 Pregnancies", 0, 15, 1)
            glucose = st.number_input("🍬 Plasma Glucose", 50, 250, 120)
            blood_pressure = st.number_input("💓 Diastolic BP (mm Hg)", 30, 120, 70)
            skin_thickness = st.number_input("🩸 Triceps Thickness (mm)", 0, 99, 20)

        with col2:
            insulin = st.number_input("💉 Serum Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("⚖️ BMI", 10.0, 50.0, 25.0)
            pedigree = st.number_input("🧬 Diabetes Pedigree", 0.0, 2.5, 0.5)
            age = st.number_input("🎂 Age", 18, 90, 30)

        # Normal prediction button
        if st.button("🚀 Predict"):
            # Create DataFrame
            input_df = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'PlasmaGlucose': glucose,
                'DiastolicBloodPressure': blood_pressure,
                'TricepsThickness': skin_thickness,
                'SerumInsulin': insulin,
                'BMI': bmi,
                'DiabetesPedigree': pedigree,
                'Age': age
            }])
            
            # Process data
            processed_data = preprocess_pipeline(input_df)
            
            # ===== DEBUG SECTION =====
            st.subheader("🔍 Debug Information")
            
            # Show processed data stats
            st.write("#### Processed Data Preview")
            st.dataframe(processed_data)
            
            # Display raw probabilities
            proba = model.predict_proba(processed_data)[0]
            st.write(f"#### Raw Probabilities")
            st.write(f"- Non-Diabetic: {proba[0]:.2%}")
            st.write(f"- Diabetic: {proba[1]:.2%}")
            # ===== END DEBUG =====
            
            # Final prediction
            prediction = model.predict(processed_data)[0]
            if prediction == 1:
                st.error("🔴 Prediction: Diabetic")
                st.warning("High diabetes risk detected. Please consult a specialist.")
            else:
                st.success("🟢 Prediction: Not Diabetic")
                st.info("No significant diabetes risk detected.")

    elif prediction_type == "Multiple Prediction":
        st.subheader("📂 Upload CSV File")
        uploaded_file = st.file_uploader("📤 Upload patient data (CSV format)", 
                                       type=["csv"],
                                       help="File should contain the same columns as single prediction inputs")
        
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.write("### Raw Patient Data Preview")
                st.dataframe(raw_df.head(3))
                
                if st.button("🔍 Predict for All Patients"):
                    with st.spinner("Processing data and making predictions..."):
                        # Process entire dataset
                        processed_df = preprocess_pipeline(raw_df.copy())
                        
                        # Get predictions
                        predictions = model.predict(processed_df)
                        results_df = processed_df.copy()
                        results_df["Prediction"] = ["Diabetic" if x == 1 else "Not Diabetic" for x in predictions]
                        results_df["Confidence"] = model.predict_proba(processed_df)[:, 1].round(2)
                        
                        st.write("### Prediction Results")
                        st.dataframe(results_df)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv,
                            file_name='diabetes_predictions.csv',
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.info("Please ensure your file matches the expected format with all required columns.")

# ---------------------- STUDY REPORT ----------------------
if menu == "📊 Study Report":
    st.title("📊 Study Report")
    
    st.write(" Here we give some insights about the data : histograms for features, class repartitions, correlation matrix between features to see relationships between them and the target variable. ")

    df = pd.read_csv("TAIPEI_diabetes.csv").drop(columns=['PatientID'])
    
    #--- 
    df.hist(bins=60, figsize=(15, 10))
    st.pyplot(plt)
    #---

    #---
    st.subheader("📌 Diabetic vs Non-Diabetic Distribution")
    #
    diabetes_counts = df['Diabetic'].value_counts()

    # ---
    plt.figure(figsize=(8, 6))
    plt.pie(diabetes_counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    st.pyplot(plt)
    st.write(" As we can see, there is an imbalance in favor of the non-diabetic class. That's not extreme, but it's there. In reality, outside of our dataset, such an imbalance is not outrageous, since diabetes only affects about 11-12% of people worldwide (cf. study).")
    #---


    #---
    st.subheader("📌 Correlation Matrix")
    # 
    correlation_matrix = df.corr()

    # ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    #---


    #---
    st.markdown("### 🔎 Insights from the Dataset (EDA)")
    eda_slides = [
        "🗂 **Dataset Overview:** 15,000 records with 8 features",
        "🛠 **Missing Values Handling:** No major missing data",
        "📊 **Feature Importance:** PlasmaGlucose and BMI are key indicators",
        "⚖️ **Diabetes Distribution:** 33.3% of the dataset individuals have diabetes",
        "🧩 **Correlation Matrix:** High correlation between Pregnancies, Age, BMI and diabetes"
    ]
    
    for slide in eda_slides:
        st.write(slide)

    #---
# ---------------------- MODEL EVALUATION ----------------------
if menu == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.write("### 🤖 Model Used: **XGBoost**")
    st.write("🏆 **Performance:** XGBoost performed best with a ROC-AUC score of **97.26%**.")

    eval_metrics = {
        "🎯 Accuracy": "93%",
        "📍 Precision": "90%",
        "🔄 Recall": "88%",
        "🧠 F1 Score": "89%"
    }
    
    for metric, value in eval_metrics.items():
        st.write(f"**{metric}:** {value}")

    st.subheader("📌 Feature Importance")
    image_url = "https://zupimages.net/up/25/13/gsap.png" 
    st.image(image_url, use_container_width=True)
    st.write("🔍 **Note:** This image represents the importance of basic and crafted (during features engineering) features in predicting diabetes using the XGBoost model.")
    #---
# ---------------------- ABOUT PAGE ----------------------
if menu == "ℹ️ About":
    st.title("ℹ️ About the Project")
    st.markdown("""
        ## 🏥 Diabetes Mellitus: A Chronic Metabolic Disorder  
        Diabetes mellitus is a condition characterized by the body's impaired ability to utilize blood sugar (glucose) effectively.  
        The **American Diabetes Association** classifies diabetes into two primary types:
        
        ### 🔹 Type 1 Diabetes (Insulin-Dependent)
        - Often manifests in **childhood**  
        - Caused by an **autoimmune response** that destroys insulin-producing beta cells  
        - The exact cause is **multifactorial**: genetic predisposition, environmental factors, and viral infections  
        
        ### 🔹 Type 2 Diabetes (Non-Insulin-Dependent)
        - More **prevalent** and typically diagnosed in **adulthood**  
        - Results from **insulin resistance** or **insufficient insulin secretion**  
        - **Risk Factors:** Family history, obesity, and physical inactivity  

        ### 🏡 Other Forms of Diabetes:
        - **Gestational Diabetes Mellitus (GDM)**: Temporary during pregnancy, increases risk of Type 2 diabetes later  
        - **Genetic Defects & Pancreatic Dysfunction**: Less common, caused by genetics or exposure to medications/chemicals  

        ---

        ## 👩‍👦 Maternal Inheritance of Diabetes:
        - 🤰 **Gestational diabetes** is unlikely to directly cause diabetes in the baby  
        - 👶 **Type 2 diabetes in the mother** increases the child's risk of Type 2 later in life  
        - 🧬 **Type 1 diabetes in the mother** slightly increases the risk of the child having Type 1 diabetes at birth  

        ---

        ## 📊 Machine Learning & Diabetes Prediction  
        - Diabetes is a **multi-factorial disease** 🏥  
        - Many **ML models** have been built to assist doctors in diagnosing diabetes  
        - The **PIMA Indian Diabetes dataset** is commonly used for research  
        - Our project is based on a **recent study**:  

        > **Chou et al., J.Pers.Med. 2023**:  
        > Study of **15,000 women (aged 20-80)** at the **Taipei Municipal Medical Center**  
        > Data collected from **2018–2020 & 2021–2022**  

        ---

        ## 📂 Dataset: TAIPEI_diabetes.csv
        **This dataset contains 15,000 records with 8 health features:**  
        - **🤰 Pregnancies:** Number of times pregnant  
        - **🍬 Plasma Glucose:** Glucose concentration after 2 hours in an oral glucose tolerance test  
        - **💓 Diastolic Blood Pressure:** Measured in mm Hg  
        - **🩸 Triceps Thickness:** Skin fold thickness (mm)  
        - **💉 Serum Insulin:** 2-Hour serum insulin (mu U/ml)  
        - **⚖️ BMI:** Body Mass Index (kg/m²)  
        - **👨‍👩‍👧 Diabetes Pedigree:** Probability of diabetes based on family history  
        - **🎂 Age:** Age in years  

        ---

    """)

    st.subheader("👨‍💻 Team Members")
    team = [
        {"name": "BRUNET Nathan", "role": "Data Scientist"},
        {"name": "IBITOWA Abraham", "role": "Data Scientist"},
        {"name": "HAOUA Anis Sofiane", "role": "Data Analyst"},
        {"name": "KAKY SUZY Joelly Magalie", "role": "Data Analyst"},
        {"name": "NIANG Falilou", "role": "Data Engineer"}
    ]

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, member in zip([col1, col2, col3, col4, col5], team):

        col.write(f"**{member['name']}**")
        col.write(member["role"])

    st.divider()
    st.write("#### 🎓 Sponsored by")
    st.image("dsti_logo.webp", width=80)  
