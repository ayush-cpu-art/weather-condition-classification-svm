import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Task 1: Load Dataset
# -------------------------------------------------

df = pd.read_csv("weather_data.csv")

print("First Five Records:\n")
print(df.head())

# -------------------------------------------------
# Create Target Variable
# -------------------------------------------------

df["Weather_Class"] = df["Temperature"].apply(
    lambda x: "Warm" if x >= 25 else "Cool"
)

print("\nDataset Shape:", df.shape)

print("\nMissing Values:\n")
print(df.isnull().sum())

# -------------------------------------------------
# Features and Target
# -------------------------------------------------

X = df[
    [
        "Temperature",
        "Relative_Humidity",
        "Surface_Pressure",
        "Wind_Speed"
    ]
]

y = df["Weather_Class"]

# -------------------------------------------------
# Encode Target Variable
# -------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# SVM Model
# -------------------------------------------------

model = SVC(
    kernel="rbf",
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------------
# Model Evaluation
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# Observations
# -------------------------------------------------

print("\nObservations")
print("1. The SVM classifier successfully classifies the weather into Warm and Cool categories.")
print("2. StandardScaler improves model performance by bringing all features to a similar scale.")
print("3. Temperature has the greatest influence because the target class is derived from it.")