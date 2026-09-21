# 🌦️ Weather Condition Classification using SVM

## 📌 Overview

This project uses a Support Vector Machine (SVM) to classify weather conditions into **Warm** and **Cool** categories based on weather-related attributes.

The classification target is created from temperature, with **25°C** used as the threshold.

---

## 🎯 Objective

Develop an SVM classifier to predict whether the weather is **Warm** or **Cool** using selected weather attributes.

---

## 📊 Dataset

The dataset contains the following features:

- Temperature
- Relative Humidity
- Surface Pressure
- Wind Speed

### Target Variable

A `Weather_Class` target is created based on temperature:

- **Warm** → Temperature ≥ 25°C
- **Cool** → Temperature < 25°C

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn

---

## ⚙️ Machine Learning Workflow

1. Load the weather dataset.
2. Preprocess the data.
3. Create the target variable.
4. Encode categorical data.
5. Split the dataset into training and testing sets using an 80:20 ratio.
6. Scale features using `StandardScaler`.
7. Train an SVM using the RBF kernel.
8. Evaluate the classification performance.

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📂 Project Structure

```text
Weather-Condition-Classification/
│
├── weather_condition_classification_svm.py
├── weather_data.csv
├── README.md
├── requirements.txt
└── .gitignore