import streamlit as st

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_recommendations(severity, bmi_category=None):
    recommendations = {}

    # ---------------- Lifestyle ----------------
    if severity == "Low":
        recommendations["Lifestyle"] = [
            "Maintain a consistent daily routine with balanced sleep and activity.",
            "Do regular moderate exercise (walking, yoga, cycling, or swimming).",
            "Practice mindfulness, meditation, or deep breathing to reduce stress.",
            "Avoid smoking and limit alcohol consumption.",
            "Track menstrual cycles using a journal or app."
        ]
    elif severity == "Medium":
        recommendations["Lifestyle"] = [
            "Aim for weight reduction of 5–10% if overweight; it significantly improves PCOS symptoms.",
            "Incorporate both cardio (brisk walking, running) and strength training.",
            "Practice sleep hygiene — maintain a consistent bedtime, avoid screens late at night.",
            "Join a support group or counseling if stress/anxiety is high.",
            "Monitor hormone levels every 6–12 months."
        ]
    elif severity == "High":
        recommendations["Lifestyle"] = [
            "Seek guidance from a gynecologist, endocrinologist, or nutritionist regularly.",
            "Exercise under medical supervision — structured workouts at least 5 days/week.",
            "Practice stress reduction: yoga, therapy, or relaxation exercises.",
            "Avoid sedentary lifestyle — take frequent breaks if working long hours.",
            "Closely monitor vital signs: blood pressure, glucose, cholesterol."
        ]

    # ---------------- Diet ----------------
    if bmi_category:
        if bmi_category == "Underweight":
            recommendations["Diet"] = [
                "Eat every 3–4 hours with calorie-dense but nutritious foods.",
                "Include healthy fats: olive oil, nuts, seeds, avocado, peanut butter.",
                "Drink milkshakes or smoothies with fruits, nut butter, or oats.",
                "Ensure adequate protein intake (lentils, paneer, eggs, dairy).",
                "Avoid empty-calorie foods like chips, sweets, or sodas."
            ]
        elif bmi_category == "Normal":
            recommendations["Diet"] = [
                "Maintain a Mediterranean-style balanced diet (whole grains, fruits, veggies, healthy fats).",
                "Choose low-GI fruits like berries, guava, papaya, apples.",
                "Prefer whole wheat, oats, quinoa instead of white rice/bread.",
                "Limit caffeine to 1–2 cups/day and avoid sugary drinks.",
                "Ensure adequate hydration (2–3 liters water daily)."
            ]
        elif bmi_category == "Overweight":
            recommendations["Diet"] = [
                "Follow a high-fiber diet: flaxseeds, leafy greens, whole grains.",
                "Eat lean proteins (paneer, legumes, beans, chicken/fish if non-veg).",
                "Avoid refined carbs (white bread, cakes, sweets).",
                "Plan smaller, frequent meals to control appetite.",
                "Replace fried foods with baked or steamed options."
            ]
        elif bmi_category == "Obese":
            recommendations["Diet"] = [
                "Strictly follow a calorie-controlled, low-GI, high-protein diet.",
                "Incorporate omega-3-rich foods: walnuts, chia, flaxseeds, fish (if non-veg).",
                "Replace juices with whole fruits for fiber and satiety.",
                "Avoid processed foods, bakery items, and sugary snacks completely.",
                "Consider medical nutrition therapy with a dietitian."
            ]

    # ---------------- Medical ----------------
    if severity == "Low":
        recommendations["Medical"] = [
            "No immediate medication required unless other symptoms develop.",
            "Focus on prevention with lifestyle and diet.",
            "Schedule a gynecology check-up once a year for monitoring."
        ]
    elif severity == "Medium":
        recommendations["Medical"] = [
            "Consult a reproductive endocrinologist or a gynecologist for advanced management.",
            "Oral contraceptives may be prescribed to regulate cycles.",
            "Metformin may be prescribed if insulin resistance is detected.",
            "Vitamin D and B12 supplementation if deficient.",
            "Regular monitoring of blood sugar and lipid profile."
        ]
    elif severity == "High":
        recommendations["Medical"] = [
            "Consult a reproductive endocrinologist or a gynecologist for advanced management.",
            "Anti-androgen medications (e.g., Spironolactone) may help with excess facial or body hair/acne.",
            "Metformin may be given by the doctor to manage insulin resistance and improve fertility chances.",
            "Fertility treatment options (Clomiphene, Letrozole, IVF) if pregnancy is planned.",
            "Regular monitoring: hormone profile, ultrasound scans, metabolic screening."
        ]

    return recommendations


# -------------------------------
# 🔹 Streamlit UI Component
# -------------------------------
st.title("PCOS Recommendations System")

# Example Inputs (replace with ML predictions later)
predicted_severity = st.selectbox("Select PCOS Severity:", ["Select","Low", "Medium", "High"], index = 0)
bmi_category = st.selectbox("Select BMI Category:", ["Select","Underweight", "Normal", "Overweight", "Obese"], index = 0)

# Get recommendations
recs = get_recommendations(predicted_severity, bmi_category)

# 🔹 User chooses which recommendation type to view
st.subheader("Personalized Recommendations")
selected_category = st.selectbox("Choose Recommendation Type:", list(recs.keys()) + ["Show All"])

# Show recommendations
if selected_category == "Show All":
    for category, advice in recs.items():
        st.markdown(f"### {category} Recommendations")
        for r in advice:
            st.write(f"- {r}")
else:
    st.markdown(f"### {selected_category} Recommendations")
    for r in recs[selected_category]:
        st.write(f"- {r}")
