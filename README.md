# Credit Risk Prediction App

## Live Demo
[Click here to open the app](https://predicting-credit-risk-for-loan-applicants-by-amaan-ali.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predicting-credit-risk-for-loan-applicants-by-amaan-ali.streamlit.app/)

This project is a **machine learning web application** that predicts whether a loan applicant is a **Good Credit Risk** or a **Bad Credit Risk** based on their financial and personal details.

Built with **Streamlit**, **scikit-learn**, and the **German Credit Data** dataset.

---

## Project Overview

- Developed a machine learning model to classify loan applicants into Good or Bad credit risk.
- Performed data preprocessing, exploratory data analysis (EDA), feature engineering, and model development.
- Deployed the model using a sleek and professional Streamlit web application.
- Used Random Forest Classifier with hyperparameter tuning for best performance.

---

## Technologies Used

- **Python 3.12**
- **Pandas, NumPy** - Data Handling
- **Matplotlib, Seaborn** - Visualization
- **Scikit-learn** - Machine Learning
- **Streamlit** - Web Application
- **Joblib** - Model Serialization

---

## Dataset

- **Name:** German Credit Data
- **Source:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/german-credit)
- **Description:** Contains attributes related to credit, personal, and employment details of individuals.

---

## How It Works

1. User enters applicant information (Age, Sex, Job type, Housing status, etc.).
2. The data is processed using the trained preprocessing pipeline.
3. The model predicts the Credit Risk: 
    - ✅ Good Credit Risk
    - ⚠️ Bad Credit Risk
4. Result is displayed with a clean, dark-themed UI.

---

## Project Structure

```bash
.
├── credit_risk_model.pkl           # Trained Random Forest Model
├── column_transformer.pkl          # Preprocessing Pipeline
├── german_credit_data.csv          # Cleaned Dataset
├── model_app.py                    # Streamlit App
└── README.md                       # Project Documentation
