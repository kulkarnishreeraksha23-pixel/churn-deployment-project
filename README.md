# Customer Churn Prediction - End-to-End AI Deployment

## Project Overview

This project is an End-to-End AI Deployment system for predicting customer churn.  
The model predicts whether a customer is likely to leave a service based on customer details such as tenure, contract type, payment method, internet service, monthly charges, and support services.

The project includes:

- Machine Learning model training
- Model serialization using Joblib
- FastAPI backend API
- Streamlit frontend interface
- Docker containerization
- API testing using Swagger/Postman

---

## Problem Statement

Customer churn is a major problem for subscription-based companies.  
By predicting churn in advance, companies can identify high-risk customers and take preventive actions such as offering discounts, better service, or customer support.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Docker

---

## Project Structure

```text
churn-deployment-project/
│
├── app/
│   ├── main.py
│   └── frontend.py
│
├── model/
│   └── churn_model.pkl
│
├── train_model.py
├── requirements.txt
├── Dockerfile
├── README.md
└── sample_request.json