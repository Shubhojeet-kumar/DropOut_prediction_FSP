# DropOut_prediction_FSP
🎓 Student Dropout Prediction System
This project is a Machine Learning solution designed to predict whether a student is likely to Graduate or Dropout based on various demographic, socio-economic, and academic factors. The goal is to identify at-risk students early so educational institutions can intervene and provide support.

📌 Project Overview
Problem Statement: High dropout rates in higher education institutions lead to resource wastage and negative career impacts for students.

Solution: A classification model that analyzes student data to predict their academic outcome.

Best Model: Logistic Regression (achieved ~91% accuracy).

📂 Dataset
The dataset contains data from a higher education institution, including:

Demographics: Age, Gender, Marital Status, Nationality.

Socio-Economic: Parents' qualification/occupation, Scholarship status, Debtor status.

Academic Info: Curricular units enrolled/approved/graded, Course type.

Target Variable: Target (Graduate / Dropout).

Note: The dataset excludes students currently "Enrolled" to focus on the binary classification of completion vs. dropout.

🛠️ Tech Stack
Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Learn, XGBoost

Model Saving: Joblib

