import streamlit as st
import numpy as np
import joblib

# Load model files
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI Crop Recommendation System")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)")

with col2:
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):

    data = np.array([[N, P, K,
                      temperature,
                      humidity,
                      ph,
                      rainfall]])

    scaled = scaler.transform(data)

    probs = model.predict_proba(scaled)[0]

    top_idx = np.argsort(probs)[::-1][:3]

    crops = encoder.inverse_transform(top_idx)

    scores = probs[top_idx] * 100

    soil_health = (
        (N/140)*25 +
        (P/145)*20 +
        (K/205)*20 +
        (1 - abs(ph-7)/7)*15 +
        (humidity/100)*10 +
        (rainfall/300)*10
    )

    soil_health = min(round(soil_health), 100)

    st.success(f"Soil Health Score: {soil_health}")

    st.subheader("Top Suitable Crops")

    st.write(f"🥇 {crops[0]} : {scores[0]:.2f}%")
    st.write(f"🥈 {crops[1]} : {scores[1]:.2f}%")
    st.write(f"🥉 {crops[2]} : {scores[2]:.2f}%")

    st.subheader("Recommended Crop")
    st.success(crops[0])

    st.subheader("Recommendations")

    if soil_health >= 80:
        st.info("Maintain current soil conditions.")

    elif soil_health >= 60:
        st.info("Apply balanced fertilizers and monitor moisture.")

    else:
        st.warning("Improve soil fertility using organic manure.")