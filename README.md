<div align="center">

  <img src="https://zupimages.net/up/25/13/rkqb.png" alt="logo" width="200" height="auto" />
  <h1>Diabetes Prediction Pipeline & Application</h1>
  
  <p>
    Predict automatically the diagnosis of diabetes. 
  </p>
  
<!-- Badges -->
<p>
  <a href="https://ml-project-diabetes-prediction-au8e37mgywogwtu4bqgymv.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
  </a>
  <a href="https://colab.research.google.com/github/NathanBrunet/ML-Project-Diabetes-Prediction/blob/main/DiabetesPrediction.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" />
  </a>
</p>
  </div>

<!-- About the Project -->
## :star2: About the Project

<!-- Features -->
### :dart: General Structure

We use the TAIPEI_diabetes dataset to train ML models within a pipeline (Jupyter Notebook). 
The best model is selected and saved, then integrated within a Streamlit-hosted application to infer diabetes predictions based on user input.

### :robot: ML Pipeline 

#### Data Preprocessing

The dataset was first loaded and then thoroughly cleaned. 
The dataset has 15,000 entries, 9 initial features (PatientID, Pregnancies, PlasmaGlucose, DiastolicBloodPressure,	TricepsThickness,	SerumInsulin,	BMI, DiabetesPedigree,	Age) for each patient corresponding to real gathered medical data, and 1 Diabetic categorical variable (1 = Diabetic/ 0 = Non-Diabetic).

The cleaning steps included:

- Check for NaN presence (Not A Number values)
- Removal of duplicates
- Removal of PatientID column which brings no valuable information

#### Exploratory Data Analysis (EDA)

A comprehensive EDA was performed, with steps including:

- Checking for missing data to identify abnormal null values (for DiastolicBloodPressure for instance, which cannot be 0)
- Distribution analysis of features (histograms, Kernel Density Estimation, boxplots) and categorical variable
- Removal of outliers based on actual medical values when it as judged necessary
- Plotting a first correlation matrix to investigate relations between features and target variable 

Examination of the average scores by essay set, highlighting the variation in grading across different topics.

#### Feature Engineering

To enrich the model's input data, several feature engineering steps were conducted:

- Binning of numerical features into 5 new categorical data features to more precise insights (Age, BMI, Glucose, BloodPressure, Pregnancies)
- Creation of 3 new indexes (Pseudo Matsuda Index, BMI_DiabetesPedigree, DBP_Age_Freq_Score)
- Visualization of the new features impact (mutual_info, Correlation Matrix) with a first manual removal of the less significative features
- Applying a Yeo-Johnson transformation and standardization of the remaining features to equalize scales and skewness

#### Model Training and Selection

For the modeling, training and testing phase:

- 5 models were investigated (Logistic Regression, Decision Tree, Random Forest, LightGBM, XGBoost) and trained using a 3-fold Recursive Feature Elimination with Cross-Validation and GridSearch for hyperparameters optimization
- Test results according to the ROC-AUC score metric and classification report data (confusion matrix, F1-score, recall, accuracy, precision) are printed for each model-
- The best model (XGBoost) according to our principal metric is then saved (with pickle module) for future use in the Streamlit application

### 💻 Streamlit Application

The Diabetes Prediction app proposes 4 pages to explore : 
- Prediction
- Study Report
- Model Evaluation
- About

#### Prediction

Allows the user to conduct a diabetes diagnosis prediction based on its inputs for the different concerned features (Age, BMI, etc.), for a single (Single Prediction button) or multiple (Multiple Prediction) individuals (by charging a CSV file), and debuging insights.

#### Study Report

Provides several key visual insights about the dataset used and its features (features histograms, class repartition - Diabetic VS Non-Diabetic, correlation matrix between features and target variable).

#### Model Evaluation

Provides statistics from the classification report of the selected model (XGBoost) used for prediction (Accuracy, Precision, Recall, F1-Score, features importances for this model, both basic and crafted during features engineering step before the model training

#### About

Provides general informations about diabetes, recap info concerning the dataset, and contributors info.

<!-- Getting Started -->
## 	:toolbox: Getting Started

<!-- Running the project online -->
### :technologist: Running the project online

Use the Streamlit badge at the beginning of the file to make automatic diabetes diagnosis predictions.

<!-- Running the project locally -->
### :exclamation: Prerequisites - Running the project locally

Clone the project
```bash
  git clone https://github.com/NathanBrunet/ML-Project-Diabetes-Prediction.git
```
Install all necessary requirements

```bash
  pip install -r requirements_full.txt
```

Go to the project directory

```bash
  cd projects/ML_Project_Diabetes_Prediction
```

Run the ML pipeline to see how it works and save automatically the best ML model from its training on the initial TAIPEI_diabetes.csv file
```bash
jupyter notebook notebook.ipynb
```

Run the Streamlit app which automatically retrieves the saved ML model to make diagnosis predictions locally and benefit from dataset insights and information
```bash
  streamlit run st_app.py
```

A small script is finally furnished to gather insights about the model used in the application. Run this command to obtain them
```bash
  python model_inspection.py
```

<!-- Contributing -->
## :wave: Contributing

- abraham.ibitowa@edu.dsti.institute
- nathan.brunet@du.dsti.institute
- anis-soufiane.haoua@edu.dsti.institute
- joelly-magalie.kaky-suzy@edu.dsti.institute
- falilou.niang@edu.dsti.institute


<!-- License -->
## :warning: License

Distributed under the MIT License. See LICENSE.txt for more information.


<!-- Contact -->
## :handshake: Contact

Nathan Brunet - nathan.brunet1343@gmail.com

Project Link: [https://github.com/NathanBrunet/ML-Project-Diabetes-Prediction](https://github.com/NathanBrunet/ML-Project-Diabetes-Prediction)


<!-- Acknowledgments -->
## :gem: Acknowledgements

This project was made possible thanks to the following tools and libraries:

- [Scikit-learn](https://scikit-learn.org/) - Machine learning algorithms  
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis  
- [Matplotlib](https://matplotlib.org/) - Data visualization  
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization  
- [Google Colab](https://colab.research.google.com/) - Cloud-based notebook execution  
- [Streamlit](https://streamlit.io/) - Interactive web app framework  
- [GitHub](https://github.com/) - Version control and project hosting  
- [Shields.io](https://shields.io/) - Badges for README customization  
- [Emoji Cheat Sheet](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md) - Emojis for README customization

