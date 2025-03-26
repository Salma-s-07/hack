import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('pcos_model.pkl')
scaler = joblib.load("scaler.pkl")

st.title("PCOS/PCOD Detection App")
st.write("Enter your symptoms to check your risk.")

# User inputs
irregular_periods = st.selectbox("Irregular Periods?", ["No", "Yes"])
weight_gain = st.selectbox("Significant Weight Gain?", ["No", "Yes"])
hair_growth = st.selectbox("Excessive Hair Growth?", ["No", "Yes"])
hormonal_imbalance = st.selectbox("Hormonal Imbalance?", ["No", "Yes"])
insulin_resistance = st.selectbox("Insulin Resistance?", ["No", "Yes"])
family_history = st.selectbox("Family History of PCOS?", ["No", "Yes"])

# Convert inputs to numerical
input_data = [
    1 if irregular_periods == "Yes" else 0,
    1 if weight_gain == "Yes" else 0,
    1 if hair_growth == "Yes" else 0,
    1 if hormonal_imbalance == "Yes" else 0,
    1 if insulin_resistance == "Yes" else 0,
    1 if family_history == "Yes" else 0
]

# Predict button
if st.button("Check PCOS Risk"):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Normalize input
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔴 High Risk of PCOS Detected!")
        st.write("🔗 [Consult an Apollo Doctor Now](https://www.apollo247.com/)")
    else:
        st.success("🟢 No signs of PCOS detected!")
