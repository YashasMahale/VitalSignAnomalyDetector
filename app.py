import streamlit as st
import requests

st.title("AI Vital Sign Monitor")

hr = st.slider("Heart Rate", 40, 180, 75)
spo2 = st.slider("SpO2", 70, 100, 98)
bp = st.slider("Blood Pressure", 80, 200, 120)

if st.button("Check Anomaly"):
    try:
        response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "hr": hr,
        "spo2": spo2,
        "bp": bp
    }
)
        result = response.json()
        
        if result["anomaly"]:
            st.error("🚨 Anomaly Detected")
        else:
            st.success("✅ Normal")
    except Exception as e:
        st.error(f"Error: {e}")
