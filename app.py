import streamlit as st


st.set_page_config(page_title="PCOS Risk Prediction", page_icon="🩺", layout="centered")

# css
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------
# Home Page
# ---------------------------
st.title("PCOS Risk Prediction System")
st.subheader("Welcome!")

st.write("""
This application helps in predicting **PCOS Risk Severity** (Low, Medium, High) 
based on medical features such as age, BMI, hormonal levels, and ovarian parameters.

### Navigation
- **EDA Page** → Exploratory Data Analysis performed on the dataset
- **Model Page** → Train models (Decision Tree) on the dataset.  
- **Prediction Page** → Enter patient details to predict risk severity.  
- **Recommendation Page** → Gives recommendations based on the risk severity.  

---

### About
This project was developed as part of **ML Unique Project**.  
It demonstrates how machine learning can be applied in the **medical domain** to assist in diagnosis and risk classification.
""")

# Motivational note
st.success("Select a page from the sidebar to get started 🚀")
