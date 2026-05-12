import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# -------------------------------
# 1. Load trained model
# -------------------------------
model = joblib.load("model/churn_model.pkl")


# -------------------------------
# 2. Create FastAPI app
# -------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An End-to-End AI Deployment project using FastAPI",
    version="1.0"
)


# -------------------------------
# 3. Input data format
# -------------------------------
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -------------------------------
# 4. Health check endpoint
# -------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Churn Prediction API is running"
    }


# -------------------------------
# 5. Home endpoint
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict"
    }


# -------------------------------
# 6. Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # Probability of churn = probability of class "Yes"
    class_labels = list(model.classes_)
    churn_index = class_labels.index("Yes")
    churn_probability = probabilities[churn_index]

    result = "Churn" if prediction == "Yes" else "Not Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(churn_probability), 4)
    }