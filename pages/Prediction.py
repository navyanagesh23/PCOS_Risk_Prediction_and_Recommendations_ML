import pandas as pd
import joblib
import streamlit as st

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load trained Random Forest model and scaler
model = joblib.load("decision_tree_model.pkl")   # Save trained model 
scaler = joblib.load("scaler.pkl")              # Save scaler too

st.title("🔮 PCOS Risk Severity Prediction")

st.write("Enter patient details below to predict **Risk Severity (Low / Medium / High)**")

# Input fields for features 
import streamlit as st

age = st.number_input("Age", min_value=15, max_value=50, value=None, placeholder="Enter Age")
bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=None, format="%.2f", placeholder="Enter BMI")
testosterone = st.number_input("Testosterone Level", min_value=15, max_value=70, value=None, placeholder="Enter Testosterone Level")
# Menstrual Irregularity with "Select" as default
menstrual_irregularity = st.selectbox(
    "Menstrual Irregularity",
    ["Select", "Yes", "No"],
    index=0 
)


antral_follicle_count = st.number_input(
    "Antral Follicle Count", 
    min_value=0, 
    max_value=50, 
    value=None,   # blank by default
    placeholder="Enter count"
)
# Convert inputs into DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Menstrual_Irregularity": [1 if menstrual_irregularity == "Yes" else 0],
    "Testosterone_Level": [testosterone],
    "Antral_Follicle_Count": [antral_follicle_count]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Risk Severity"):
    prediction = model.predict(input_scaled)[0]
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"🩺 Predicted Risk Severity: **{mapping[prediction]}**")
