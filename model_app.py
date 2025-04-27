import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and ColumnTransformer
best_rf = joblib.load('credit_risk_model.pkl')
ct = joblib.load('column_transformer.pkl')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered", initial_sidebar_state="expanded")

# --- STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    .stTextInput>div>div>input {
        background-color: #20232a;
        color: white;
    }
    .stNumberInput>div>div>input {
        background-color: #20232a;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #20232a;
        color: white;
    }
    .stButton>button {
        background-color: #0099ff;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #007acc;
        color: white;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- APP TITLE ---
st.title("💳 Credit Risk Prediction App")
st.markdown("### Predict if a loan applicant is a **Good** or **Bad** credit risk.")

st.write("---")  # separator

# --- INPUT FORM ---

st.subheader("📋 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    job = st.number_input('Job Type (0-3)', min_value=0, max_value=3)
    credit_amount = st.number_input('Credit Amount', min_value=0, value=1000)
    saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'quite rich', 'rich'])

with col2:
    sex = st.selectbox('Sex', ['male', 'female'])
    housing = st.selectbox('Housing', ['own', 'free', 'rent'])
    duration = st.number_input('Loan Duration (months)', min_value=1, value=12)
    checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'rich'])

purpose = st.selectbox('Purpose', ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

# --- DATAFRAME CREATION ---
input_df = pd.DataFrame([{
    'Age': age,
    'Sex': sex,
    'Job' : job,
    'Housing': housing,
    'Saving accounts': saving_accounts,
    'Checking account': checking_account,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': purpose
}])

st.write("---")  # separator

# --- PREDICTION ---
if st.button('🔎 Predict Credit Risk'):
    with st.spinner('Analyzing... Please wait ⏳'):
        input_encoded = ct.transform(input_df)
        prediction = best_rf.predict(input_encoded)

    risk = '✅ Good Credit Risk' if prediction[0] == 0 else '⚠️ Bad Credit Risk'
    
    # LARGER FONT for result
    result_color = "lightgreen" if prediction[0] == 0 else "tomato"
    st.markdown(
        f"<h2 style='text-align: center; color: {result_color}; font-size: 36px;'>{risk}</h2>",
        unsafe_allow_html=True
    )

# --- FOOTER ---
st.write("")
st.caption("Made by AMAAN ALI using Streamlit | 2025")
