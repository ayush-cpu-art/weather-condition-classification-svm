# 🌦️ Weather Condition Classification using SVM

A machine learning project that uses a **Support Vector Machine (SVM)** to classify weather conditions into **Warm** and **Cool** categories based on weather-related attributes.

The project demonstrates a complete classification workflow including data preprocessing, feature scaling, SVM training, and model evaluation.

---

## 📌 Overview

The model predicts whether the weather condition is **Warm** or **Cool** using:

- Temperature
- Relative Humidity
- Surface Pressure
- Wind Speed

The target variable `Weather_Class` is created using a temperature threshold:

- **Warm:** Temperature ≥ 25°C
- **Cool:** Temperature < 25°C

---

## 🎯 Objective

The main objectives of this project are to:

- Analyze weather-related data.
- Create a classification target from temperature.
- Preprocess and prepare the dataset.
- Scale features using `StandardScaler`.
- Train an SVM classifier using an RBF kernel.
- Evaluate classification performance using multiple metrics.

---

## 📊 Dataset

The dataset contains weather measurements used to classify weather conditions.

### Features

| Feature | Description |
|---|---|
| Temperature | Temperature measurement in °C |
| Relative Humidity | Relative humidity level |
| Surface Pressure | Atmospheric surface pressure |
| Wind Speed | Wind speed measurement |

### Target

| Class | Condition |
|---|---|
| Warm | Temperature ≥ 25°C |
| Cool | Temperature < 25°C |

---

## 🧠 Machine Learning Workflow

```text
Weather Dataset
      ↓
Data Preprocessing
      ↓
Create Weather_Class
      ↓
Label Encoding
      ↓
Train-Test Split (80:20)
      ↓
Feature Scaling
      ↓
SVM with RBF Kernel
      ↓
Model Evaluation
