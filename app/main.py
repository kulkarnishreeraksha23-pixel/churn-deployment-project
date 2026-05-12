import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# -------------------------------------------------
# Load trained model pipeline
# -------------------------------------------------
model = joblib.load("model/churn_model.pkl")


# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction and Retention Support System",
    description="End-to-End AI Deployment Project using FastAPI, Machine Learning, Docker, and Render",
    version="2.0"
)


# -------------------------------------------------
# Input schema using Pydantic
# -------------------------------------------------
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


# -------------------------------------------------
# Helper function for risk level and recommendation
# -------------------------------------------------
def get_risk_details(probability):
    if probability >= 0.70:
        return {
            "risk_level": "High Risk",
            "risk_color": "red",
            "recommendation": "Customer has high churn risk. Suggested action: provide retention offer, priority support, and personalized follow-up."
        }
    elif probability >= 0.40:
        return {
            "risk_level": "Medium Risk",
            "risk_color": "orange",
            "recommendation": "Customer has moderate churn risk. Suggested action: monitor usage, improve engagement, and offer suitable service plans."
        }
    else:
        return {
            "risk_level": "Low Risk",
            "risk_color": "green",
            "recommendation": "Customer appears stable. Suggested action: maintain service quality and continue regular engagement."
        }


# -------------------------------------------------
# Professional website homepage
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def website():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Customer Churn Prediction System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f172a;
            color: #0f172a;
        }

        .hero {
            background: linear-gradient(135deg, #020617, #1e3a8a, #2563eb);
            color: white;
            padding: 45px 20px 70px;
            text-align: center;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            padding: 8px 18px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 15px;
        }

        .hero h1 {
            font-size: 42px;
            margin: 10px 0;
            letter-spacing: -1px;
        }

        .hero p {
            max-width: 850px;
            margin: 0 auto;
            font-size: 18px;
            line-height: 1.6;
            color: #dbeafe;
        }

        .main-wrapper {
            max-width: 1180px;
            margin: -45px auto 40px;
            padding: 0 20px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 25px;
        }

        .status-card {
            background: white;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 14px 35px rgba(15,23,42,0.25);
        }

        .status-card h3 {
            margin: 0 0 8px;
            font-size: 15px;
            color: #475569;
        }

        .status-card p {
            margin: 0;
            font-size: 22px;
            font-weight: 800;
            color: #1e3a8a;
        }

        .system-panel {
            background: white;
            border-radius: 24px;
            box-shadow: 0 18px 45px rgba(15,23,42,0.30);
            overflow: hidden;
        }

        .panel-header {
            background: #f8fafc;
            padding: 25px 30px;
            border-bottom: 1px solid #e2e8f0;
        }

        .panel-header h2 {
            margin: 0;
            color: #1e293b;
            font-size: 26px;
        }

        .panel-header p {
            margin: 8px 0 0;
            color: #64748b;
            font-size: 15px;
        }

        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 0;
        }

        .form-section {
            padding: 30px;
        }

        .result-section {
            background: #f8fafc;
            padding: 30px;
            border-left: 1px solid #e2e8f0;
        }

        .section-title {
            font-size: 18px;
            font-weight: 800;
            margin: 25px 0 15px;
            color: #1e3a8a;
        }

        .section-title:first-child {
            margin-top: 0;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .field {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;
            font-weight: 700;
            color: #334155;
            margin-bottom: 6px;
        }

        select, input {
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            background: white;
            font-size: 15px;
            color: #0f172a;
        }

        select:focus, input:focus {
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
        }

        .button-row {
            margin-top: 28px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            border-radius: 14px;
            padding: 14px 24px;
            font-size: 15px;
            font-weight: 800;
            cursor: pointer;
            transition: 0.2s ease;
        }

        .primary-btn {
            background: #2563eb;
            color: white;
        }

        .primary-btn:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
        }

        .secondary-btn {
            background: #e2e8f0;
            color: #1e293b;
        }

        .secondary-btn:hover {
            background: #cbd5e1;
        }

        .result-card {
            background: white;
            border-radius: 18px;
            padding: 24px;
            border: 1px solid #e2e8f0;
            min-height: 260px;
        }

        .result-placeholder {
            color: #64748b;
            line-height: 1.6;
            font-size: 15px;
        }

        .prediction-label {
            font-size: 16px;
            color: #64748b;
            margin-bottom: 8px;
        }

        .prediction-main {
            font-size: 30px;
            font-weight: 900;
            margin-bottom: 18px;
        }

        .probability-box {
            margin: 18px 0;
        }

        .probability-value {
            display: flex;
            justify-content: space-between;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .progress-bar {
            height: 14px;
            background: #e2e8f0;
            border-radius: 999px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            border-radius: 999px;
            transition: width 0.5s ease;
        }

        .risk-badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 800;
            margin: 12px 0;
        }

        .recommendation {
            margin-top: 18px;
            background: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 14px;
            border-radius: 10px;
            color: #334155;
            line-height: 1.5;
            font-size: 14px;
        }

        .red-text {
            color: #dc2626;
        }

        .orange-text {
            color: #ea580c;
        }

        .green-text {
            color: #16a34a;
        }

        .red-fill {
            background: #dc2626;
        }

        .orange-fill {
            background: #ea580c;
        }

        .green-fill {
            background: #16a34a;
        }

        .red-badge {
            background: #fee2e2;
            color: #991b1b;
        }

        .orange-badge {
            background: #ffedd5;
            color: #9a3412;
        }

        .green-badge {
            background: #dcfce7;
            color: #166534;
        }

        .api-box {
            margin-top: 22px;
            background: #0f172a;
            color: white;
            border-radius: 18px;
            padding: 20px;
        }

        .api-box h3 {
            margin-top: 0;
            color: #bfdbfe;
        }

        .api-box a {
            display: block;
            color: #93c5fd;
            text-decoration: none;
            margin: 10px 0;
            font-weight: 700;
        }

        .api-box a:hover {
            text-decoration: underline;
        }

        .architecture {
            background: white;
            border-radius: 22px;
            padding: 28px;
            margin-top: 28px;
            box-shadow: 0 14px 35px rgba(15,23,42,0.18);
        }

        .architecture h2 {
            margin-top: 0;
            color: #1e293b;
        }

        .flow {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 12px;
            margin-top: 18px;
        }

        .flow-step {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 15px;
            border-radius: 14px;
            text-align: center;
            color: #1e3a8a;
            font-weight: 800;
            font-size: 13px;
        }

        .footer {
            text-align: center;
            color: #cbd5e1;
            padding: 25px;
            font-size: 14px;
        }

        @media (max-width: 950px) {
            .status-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .content-grid {
                grid-template-columns: 1fr;
            }

            .result-section {
                border-left: none;
                border-top: 1px solid #e2e8f0;
            }

            .flow {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 600px) {
            .hero h1 {
                font-size: 30px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .status-grid {
                grid-template-columns: 1fr;
            }

            .form-section, .result-section {
                padding: 20px;
            }

            .flow {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

    <section class="hero">
        <div class="hero-badge">End-to-End AI Deployment Capstone</div>
        <h1>Customer Churn Prediction and Retention Support System</h1>
        <p>
            A deployed machine learning system where users can enter customer data through a live web interface
            and instantly receive churn prediction, probability score, risk level, and business recommendation.
        </p>
    </section>

    <main class="main-wrapper">

        <section class="status-grid">
            <div class="status-card">
                <h3>Model Status</h3>
                <p>Loaded</p>
            </div>
            <div class="status-card">
                <h3>Backend API</h3>
                <p>FastAPI</p>
            </div>
            <div class="status-card">
                <h3>Deployment</h3>
                <p>Render</p>
            </div>
            <div class="status-card">
                <h3>Prediction Mode</h3>
                <p>Instant</p>
            </div>
        </section>

        <section class="system-panel">
            <div class="panel-header">
                <h2>Live Churn Prediction Dashboard</h2>
                <p>Fill the customer details below. The website sends the data to the deployed FastAPI prediction endpoint and displays the result instantly.</p>
            </div>

            <div class="content-grid">

                <div class="form-section">
                    <form id="predictionForm">

                        <div class="section-title">1. Customer Profile</div>
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
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
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
                                <label>Tenure in Months</label>
                                <input type="number" id="tenure" value="12" min="1" max="72">
                            </div>
                        </div>

                        <div class="section-title">2. Service Information</div>
                        <div class="form-grid">
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
                        </div>

                        <div class="section-title">3. Billing and Contract Details</div>
                        <div class="form-grid">
                            <div class="field">
                                <label>Contract Type</label>
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

                        <div class="button-row">
                            <button type="submit" class="primary-btn">Predict Churn</button>
                            <button type="button" class="secondary-btn" onclick="loadHighRiskSample()">Load High Risk Sample</button>
                            <button type="button" class="secondary-btn" onclick="loadLowRiskSample()">Load Low Risk Sample</button>
                        </div>

                    </form>
                </div>

                <div class="result-section">
                    <div class="result-card" id="resultCard">
                        <div class="result-placeholder">
                            <strong>Prediction Result</strong><br><br>
                            Submit customer information to receive:
                            <br><br>
                            • Churn / Not Churn prediction<br>
                            • Churn probability score<br>
                            • Risk category<br>
                            • Retention recommendation
                        </div>
                    </div>

                    <div class="api-box">
                        <h3>System Links</h3>
                        <a href="/health" target="_blank">GET /health - API Health Check</a>
                        <a href="/docs" target="_blank">Swagger API Documentation</a>
                        <a href="/predict" target="_blank">POST /predict - Prediction Endpoint</a>
                    </div>
                </div>

            </div>
        </section>

        <section class="architecture">
            <h2>System Architecture</h2>
            <p>
                This project follows a complete deployment pipeline from model training to live cloud deployment.
            </p>

            <div class="flow">
                <div class="flow-step">User Input</div>
                <div class="flow-step">Web Interface</div>
                <div class="flow-step">FastAPI Backend</div>
                <div class="flow-step">Pydantic Validation</div>
                <div class="flow-step">ML Prediction</div>
                <div class="flow-step">Instant Result</div>
            </div>
        </section>

    </main>

    <footer class="footer">
        Customer Churn Prediction and Retention Support System | FastAPI + Scikit-Learn + Docker + Render
    </footer>

    <script>
        function getValue(id) {
            return document.getElementById(id).value;
        }

        function setValue(id, value) {
            document.getElementById(id).value = value;
        }

        function loadHighRiskSample() {
            setValue("gender", "Female");
            setValue("SeniorCitizen", "0");
            setValue("Partner", "Yes");
            setValue("Dependents", "No");
            setValue("tenure", "12");
            setValue("PhoneService", "Yes");
            setValue("MultipleLines", "No");
            setValue("InternetService", "Fiber optic");
            setValue("OnlineSecurity", "No");
            setValue("OnlineBackup", "Yes");
            setValue("DeviceProtection", "No");
            setValue("TechSupport", "No");
            setValue("StreamingTV", "Yes");
            setValue("StreamingMovies", "No");
            setValue("Contract", "Month-to-month");
            setValue("PaperlessBilling", "Yes");
            setValue("PaymentMethod", "Electronic check");
            setValue("MonthlyCharges", "75.5");
            setValue("TotalCharges", "900.0");
        }

        function loadLowRiskSample() {
            setValue("gender", "Male");
            setValue("SeniorCitizen", "0");
            setValue("Partner", "Yes");
            setValue("Dependents", "Yes");
            setValue("tenure", "60");
            setValue("PhoneService", "Yes");
            setValue("MultipleLines", "Yes");
            setValue("InternetService", "DSL");
            setValue("OnlineSecurity", "Yes");
            setValue("OnlineBackup", "Yes");
            setValue("DeviceProtection", "Yes");
            setValue("TechSupport", "Yes");
            setValue("StreamingTV", "No");
            setValue("StreamingMovies", "No");
            setValue("Contract", "Two year");
            setValue("PaperlessBilling", "No");
            setValue("PaymentMethod", "Credit card (automatic)");
            setValue("MonthlyCharges", "45.0");
            setValue("TotalCharges", "2700.0");
        }

        document.getElementById("predictionForm").addEventListener("submit", async function(event) {
            event.preventDefault();

            const resultCard = document.getElementById("resultCard");

            resultCard.innerHTML = `
                <div class="result-placeholder">
                    <strong>Processing request...</strong><br><br>
                    Sending customer data to deployed FastAPI backend and running model inference.
                </div>
            `;

            const data = {
                gender: getValue("gender"),
                SeniorCitizen: parseInt(getValue("SeniorCitizen")),
                Partner: getValue("Partner"),
                Dependents: getValue("Dependents"),
                tenure: parseInt(getValue("tenure")),
                PhoneService: getValue("PhoneService"),
                MultipleLines: getValue("MultipleLines"),
                InternetService: getValue("InternetService"),
                OnlineSecurity: getValue("OnlineSecurity"),
                OnlineBackup: getValue("OnlineBackup"),
                DeviceProtection: getValue("DeviceProtection"),
                TechSupport: getValue("TechSupport"),
                StreamingTV: getValue("StreamingTV"),
                StreamingMovies: getValue("StreamingMovies"),
                Contract: getValue("Contract"),
                PaperlessBilling: getValue("PaperlessBilling"),
                PaymentMethod: getValue("PaymentMethod"),
                MonthlyCharges: parseFloat(getValue("MonthlyCharges")),
                TotalCharges: parseFloat(getValue("TotalCharges"))
            };

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error("Prediction request failed");
                }

                const result = await response.json();

                const probability = (result.churn_probability * 100).toFixed(2);
                const riskLevel = result.risk_level;
                const recommendation = result.recommendation;

                let colorClass = "green";
                if (riskLevel === "High Risk") {
                    colorClass = "red";
                } else if (riskLevel === "Medium Risk") {
                    colorClass = "orange";
                }

                let predictionText = "Customer is not likely to Churn";
                if (result.prediction === "Churn") {
                    predictionText = "Customer is likely to Churn";
                }

                resultCard.innerHTML = `
                    <div class="prediction-label">Prediction Output</div>
                    <div class="prediction-main ${colorClass}-text">${predictionText}</div>

                    <div class="probability-box">
                        <div class="probability-value">
                            <span>Churn Probability</span>
                            <span>${probability}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${colorClass}-fill" style="width:${probability}%"></div>
                        </div>
                    </div>

                    <div class="risk-badge ${colorClass}-badge">${riskLevel}</div>

                    <div class="recommendation">
                        <strong>Business Recommendation:</strong><br>
                        ${recommendation}
                    </div>
                `;

            } catch (error) {
                resultCard.innerHTML = `
                    <div class="prediction-main red-text">Error</div>
                    <div class="recommendation">
                        Unable to get prediction. Please check whether the backend API is running properly.
                    </div>
                `;
            }
        });
    </script>

</body>
</html>
    """


# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Churn Prediction API is running",
        "model_status": "loaded",
        "service": "Customer Churn Prediction System"
    }


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    class_labels = list(model.classes_)
    churn_index = class_labels.index("Yes")
    churn_probability = float(probabilities[churn_index])

    result = "Churn" if prediction == "Yes" else "Not Churn"

    risk_details = get_risk_details(churn_probability)

    return {
        "prediction": result,
        "churn_probability": round(churn_probability, 4),
        "risk_level": risk_details["risk_level"],
        "recommendation": risk_details["recommendation"]
    }