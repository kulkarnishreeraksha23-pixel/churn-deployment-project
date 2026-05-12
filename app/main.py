import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("model/churn_model.pkl")


# -------------------------------
# Create FastAPI app
# -------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An End-to-End AI Deployment project using FastAPI",
    version="1.0"
)


# -------------------------------
# Input schema
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
# Website homepage
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def website():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a, #1e293b, #2563eb);
            color: #111827;
        }

        .container {
            max-width: 1100px;
            margin: 40px auto;
            background: #ffffff;
            border-radius: 18px;
            padding: 35px;
            box-shadow: 0 20px 45px rgba(0,0,0,0.25);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #1e3a8a;
            font-size: 36px;
            margin-bottom: 8px;
        }

        .header p {
            color: #475569;
            font-size: 17px;
        }

        .badge {
            display: inline-block;
            background: #dbeafe;
            color: #1d4ed8;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: bold;
            margin-bottom: 12px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }

        .field {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: bold;
            margin-bottom: 6px;
            color: #334155;
            font-size: 14px;
        }

        select, input {
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-size: 15px;
            background: #f8fafc;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #2563eb;
            background: #ffffff;
        }

        .button-box {
            text-align: center;
            margin-top: 30px;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 14px 40px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 30px;
            padding: 22px;
            border-radius: 14px;
            text-align: center;
            font-size: 20px;
            display: none;
        }

        .churn {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }

        .not-churn {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }

        .links {
            margin-top: 35px;
            text-align: center;
            color: #475569;
        }

        .links a {
            color: #2563eb;
            text-decoration: none;
            font-weight: bold;
            margin: 0 10px;
        }

        @media (max-width: 900px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .container {
                margin: 15px;
                padding: 22px;
            }

            .header h1 {
                font-size: 28px;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <div class="badge">End-to-End AI Deployment Project</div>
            <h1>Customer Churn Prediction System</h1>
            <p>Enter customer details and get instant churn prediction using a deployed machine learning model.</p>
        </div>

        <form id="predictionForm">
            <div class="form-grid">

                <div class="field">
                    <label>Gender</label>
                    <select id="gender">
                        <option>Female</option>
                        <option>Male</option>
                    </select>
                </div>

                <div class="field">
                    <label>Senior Citizen</label>
                    <select id="SeniorCitizen">
                        <option value="0">0</option>
                        <option value="1">1</option>
                    </select>
                </div>

                <div class="field">
                    <label>Partner</label>
                    <select id="Partner">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>

                <div class="field">
                    <label>Dependents</label>
                    <select id="Dependents">
                        <option>No</option>
                        <option>Yes</option>
                    </select>
                </div>

                <div class="field">
                    <label>Tenure Months</label>
                    <input type="number" id="tenure" value="12" min="1" max="72">
                </div>

                <div class="field">
                    <label>Phone Service</label>
                    <select id="PhoneService">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>

                <div class="field">
                    <label>Multiple Lines</label>
                    <select id="MultipleLines">
                        <option>No</option>
                        <option>Yes</option>
                        <option>No phone service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Internet Service</label>
                    <select id="InternetService">
                        <option>Fiber optic</option>
                        <option>DSL</option>
                        <option>No</option>
                    </select>
                </div>

                <div class="field">
                    <label>Online Security</label>
                    <select id="OnlineSecurity">
                        <option>No</option>
                        <option>Yes</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Online Backup</label>
                    <select id="OnlineBackup">
                        <option>Yes</option>
                        <option>No</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Device Protection</label>
                    <select id="DeviceProtection">
                        <option>No</option>
                        <option>Yes</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Tech Support</label>
                    <select id="TechSupport">
                        <option>No</option>
                        <option>Yes</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Streaming TV</label>
                    <select id="StreamingTV">
                        <option>Yes</option>
                        <option>No</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Streaming Movies</label>
                    <select id="StreamingMovies">
                        <option>No</option>
                        <option>Yes</option>
                        <option>No internet service</option>
                    </select>
                </div>

                <div class="field">
                    <label>Contract</label>
                    <select id="Contract">
                        <option>Month-to-month</option>
                        <option>One year</option>
                        <option>Two year</option>
                    </select>
                </div>

                <div class="field">
                    <label>Paperless Billing</label>
                    <select id="PaperlessBilling">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>

                <div class="field">
                    <label>Payment Method</label>
                    <select id="PaymentMethod">
                        <option>Electronic check</option>
                        <option>Mailed check</option>
                        <option>Bank transfer (automatic)</option>
                        <option>Credit card (automatic)</option>
                    </select>
                </div>

                <div class="field">
                    <label>Monthly Charges</label>
                    <input type="number" id="MonthlyCharges" value="75.5" step="0.01">
                </div>

                <div class="field">
                    <label>Total Charges</label>
                    <input type="number" id="TotalCharges" value="900.0" step="0.01">
                </div>

            </div>

            <div class="button-box">
                <button type="submit">Predict Churn</button>
            </div>
        </form>

        <div id="resultBox" class="result"></div>

        <div class="links">
            <p>API Endpoints:</p>
            <a href="/health" target="_blank">Health Check</a>
            <a href="/docs" target="_blank">Swagger API Docs</a>
        </div>
    </div>

    <script>
        document.getElementById("predictionForm").addEventListener("submit", async function(event) {
            event.preventDefault();

            const data = {
                gender: document.getElementById("gender").value,
                SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value),
                Partner: document.getElementById("Partner").value,
                Dependents: document.getElementById("Dependents").value,
                tenure: parseInt(document.getElementById("tenure").value),
                PhoneService: document.getElementById("PhoneService").value,
                MultipleLines: document.getElementById("MultipleLines").value,
                InternetService: document.getElementById("InternetService").value,
                OnlineSecurity: document.getElementById("OnlineSecurity").value,
                OnlineBackup: document.getElementById("OnlineBackup").value,
                DeviceProtection: document.getElementById("DeviceProtection").value,
                TechSupport: document.getElementById("TechSupport").value,
                StreamingTV: document.getElementById("StreamingTV").value,
                StreamingMovies: document.getElementById("StreamingMovies").value,
                Contract: document.getElementById("Contract").value,
                PaperlessBilling: document.getElementById("PaperlessBilling").value,
                PaymentMethod: document.getElementById("PaymentMethod").value,
                MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
                TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
            };

            const resultBox = document.getElementById("resultBox");
            resultBox.style.display = "block";
            resultBox.className = "result";
            resultBox.innerHTML = "Predicting...";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                const probability = (result.churn_probability * 100).toFixed(2);

                if (result.prediction === "Churn") {
                    resultBox.className = "result churn";
                    resultBox.innerHTML = "⚠️ Prediction: Customer is likely to Churn<br><br>Churn Probability: " + probability + "%";
                } else {
                    resultBox.className = "result not-churn";
                    resultBox.innerHTML = "✅ Prediction: Customer is not likely to Churn<br><br>Churn Probability: " + probability + "%";
                }

            } catch (error) {
                resultBox.className = "result churn";
                resultBox.innerHTML = "Error: Unable to get prediction. Please try again.";
            }
        });
    </script>
</body>
</html>
    """


# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Churn Prediction API is running"
    }


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    class_labels = list(model.classes_)
    churn_index = class_labels.index("Yes")
    churn_probability = probabilities[churn_index]

    result = "Churn" if prediction == "Yes" else "Not Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(churn_probability), 4)
    }