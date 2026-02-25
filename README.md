VitalSignAnomalyDetector is a full-stack machine learning system designed to monitor healthcare vitals such as:

- Heart Rate (HR)
- Oxygen Saturation (SpO2)
- Blood Pressure (BP)

The system uses an LSTM Autoencoder trained on normal time-series data to detect anomalies based on reconstruction error. It is deployed via a FastAPI backend and monitored through an interactive Streamlit dashboard.

Streamlit UI -> FastAPI Backend -> LSTM Model -> Anomaly Detection -> JSON Response

- **Frontend**: Streamlit dashboard for real-time monitoring
- **Backend**: FastAPI REST API for inference
- **Model**: TensorFlow 2 LSTM Autoencoder
- **Anomaly Logic**: Statistical thresholding
- **Optional Extension**: Isolation Forest hybrid detection

To run locally use this in terminal:
git clone https://github.com/YashasMahale/VitalSignAnomalyDetector.git  
cd VitalSignAnomalyDetector
