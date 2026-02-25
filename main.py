from fastapi import FastAPI
from pydantic import BaseModel
from model import build_model, train_model
from detector import lstm_check

app = FastAPI()

model = build_model()
model, threshold = train_model(model)

class Vitals(BaseModel):
    hr: float
    spo2: float
    bp: float

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(vitals: Vitals):
    window = [[vitals.hr, vitals.spo2, vitals.bp]] * 100
    anomaly = lstm_check(model, threshold, window)
    return {"anomaly": anomaly}
