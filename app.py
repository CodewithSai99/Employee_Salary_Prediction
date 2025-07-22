import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Model and Columns
# -----------------------------
model = joblib.load("best_model.pkl")
input_columns = joblib.load("input_columns.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")

# User Inputs
company = st.selectbox("Select Company", ['Google', 'Amazon', 'Facebook', 'Microsoft', 'TCS', 'Infosys', 'Wipro'])
education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
location = st.selectbox("Location", ['Bangalore', 'Hyderabad', 'Delhi', 'Mumbai', 'Chennai', 'Pune'])
job_title = st.selectbox("Job Title", ['Data Scientist', 'Software Engineer', 'Product Manager', 'HR', 'Business Analyst'])

# Predict button
if st.button("🔍 Predict Salary"):
    # Build user input dictionary
    user_input = {
        'Company': company,
        'Education Level': education,
        'Years of Experience': experience,
        'Location': location,
        'Job Title': job_title,
    }

    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)

    # Add any missing columns
    for col in input_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[input_columns]

    # Predict
    predicted_salary = model.predict(input_df)
    st.success(f"💰 Predicted Salary: ₹{predicted_salary[0]:,.2f}")

