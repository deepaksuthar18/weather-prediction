import pickle

import numpy as np
import streamlit as st

# Load Models
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("🌦️ Weather Prediction App")
st.write("Enter the weather details to predict Temperature and Weather Condition")

# User Inputs
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
temp_min = st.number_input("Minimum Temperature (°C)", step=0.1)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
year = st.number_input("Year", min_value=2000, step=1)
month = st.slider("Month", 1, 12, 1)
day = st.slider("Day", 1, 31, 1)

# Predict Button
if st.button("Predict Weather"):
    # Preprocess Input
    input_data = np.array([[precipitation, temp_min, wind, year, month, day]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = best_model.predict(input_scaled)

    # Display Result
    if hasattr(best_model, 'predict_proba'):
        st.success(f"Predicted Weather Condition: {int(prediction[0])}")
    else:
        st.success(f"Predicted Maximum Temperature: {round(prediction[0], 2)}°C")
