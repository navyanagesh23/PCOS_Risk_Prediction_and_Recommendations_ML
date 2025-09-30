import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_data():
    df = pd.read_csv("Revised_Pcos_Dataset.csv")
    return df

df = load_data()

st.title("PCOS Risk Severity Prediction & Recommendations")
st.write(" Note: The data used in this project is collected after medical examinations. Hence, it includes clinical parameters such as testosterone levels and antral follicle count, which are important for assessing conditions like PCOS. This dataset is strictly for academic/research purposes.")


st.write("Step 1: Load and Explore the dataset")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------
st.header("Exploratory Data Analysis (EDA)")

import io

# Dataset Info
if st.checkbox("Show Dataset Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

# Summary Statistics
if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

# Null Values
if st.checkbox("Show Null Values"):
    st.write(df.isnull().sum())

# ---------------------------------
# Feature Distributions
# ---------------------------------
st.subheader("Feature Distributions")

numeric_features = ["Age", "BMI", "Testosterone_Level", "Antral_Follicle_Count"]
categorical_features = ["Menstrual_Irregularity", "Risk_Severity"]

# Distribution of Numeric Features
st.subheader("📈 Feature Distributions (Numeric)")
numeric_features = ["Age", "BMI", "Testosterone_Level", "Antral_Follicle_Count"]
colors = ["red", "blue", "green", "orange"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(numeric_features):
    row, col = i // 2, i % 2
    sns.histplot(df[feature], bins=20, kde=True, ax=axes[row, col], color=colors[i])
    axes[row, col].set_title(f"{feature} Distribution", fontsize=12)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
🔎 **Interpretation:**  
- **Age**: Common age range of patients.  
- **BMI**: Body weight distribution, higher BMI may link to higher PCOS risk.  
- **Testosterone**: Elevated values could indicate hormonal imbalance.  
- **Follicle Count**: Higher counts may relate to PCOS symptoms.  
""")

# Distribution of Categorical Features
st.subheader("📊 Categorical Feature Distributions")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x="Menstrual_Irregularity", data=df, ax=axes[0], palette="Pastel1")
axes[0].set_title("Menstrual Irregularity")
sns.countplot(x="Risk_Severity", data=df, ax=axes[1], palette="Set2")
axes[1].set_title("Risk Severity")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
🔎 **Interpretation:**  
- **Menstrual Irregularity**: How many patients have irregular vs regular cycles.  
- **Risk Severity**: Shows target class balance (Low, Medium, High).  
""")

# Risk Severity Pie Chart (Class Balance)
st.subheader("🎯 Risk Severity Class Balance")
risk_counts = df["Risk_Severity"].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%",
       colors=["#FF9999", "#66B2FF", "#99FF99"], startangle=90, explode=(0.05, 0.05, 0.05))
ax.set_title("Risk Severity Distribution")
st.pyplot(fig)



# Correlation Heatmap
st.subheader("🔗 Correlation Heatmap (Numeric Features)")
corr = df[numeric_features].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)


