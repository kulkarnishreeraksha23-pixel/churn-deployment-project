import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Create model folder if not present
os.makedirs("model", exist_ok=True)


# -------------------------------
# 1. Create sample churn dataset
# -------------------------------
def create_sample_dataset():
    np.random.seed(42)

    data = {
        "gender": np.random.choice(["Male", "Female"], 500),
        "SeniorCitizen": np.random.choice([0, 1], 500),
        "Partner": np.random.choice(["Yes", "No"], 500),
        "Dependents": np.random.choice(["Yes", "No"], 500),
        "tenure": np.random.randint(1, 72, 500),
        "PhoneService": np.random.choice(["Yes", "No"], 500),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], 500),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], 500),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], 500),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], 500),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], 500),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], 500),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], 500),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], 500),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 500),
        "PaperlessBilling": np.random.choice(["Yes", "No"], 500),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            500,
        ),
        "MonthlyCharges": np.random.uniform(20, 120, 500).round(2),
    }

    df = pd.DataFrame(data)

    # Create TotalCharges based on tenure and monthly charges
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)

    # Simple logical churn rule for sample data
    churn_probability = (
        (df["Contract"] == "Month-to-month").astype(int) * 0.35
        + (df["InternetService"] == "Fiber optic").astype(int) * 0.20
        + (df["TechSupport"] == "No").astype(int) * 0.15
        + (df["PaymentMethod"] == "Electronic check").astype(int) * 0.15
        + (df["tenure"] < 12).astype(int) * 0.15
    )

    random_values = np.random.random(500)
    df["Churn"] = np.where(random_values < churn_probability, "Yes", "No")

    return df


# -------------------------------
# 2. Load data
# -------------------------------
df = create_sample_dataset()

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())


# -------------------------------
# 3. Split input and output
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]


# -------------------------------
# 4. Define columns
# -------------------------------
categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


# -------------------------------
# 5. Preprocessor
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ]
)


# -------------------------------
# 6. Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# -------------------------------
# 7. Pipeline
# -------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# -------------------------------
# 8. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 9. Train model
# -------------------------------
pipeline.fit(X_train, y_train)


# -------------------------------
# 10. Evaluate model
# -------------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -------------------------------
# 11. Save model pipeline
# -------------------------------
joblib.dump(pipeline, "model/churn_model.pkl")

print("\nModel saved successfully at: model/churn_model.pkl")