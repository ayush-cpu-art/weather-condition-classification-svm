# 🌦️ Weather Condition Classification using SVM

This project classifies weather conditions as **Warm** or **Cool** using a Support Vector Machine (SVM).

## Objective

Develop an SVM classifier using weather attributes to predict whether the weather is Warm or Cool.

---

## Dataset

The dataset contains the following features:

- Temperature
- Relative Humidity
- Surface Pressure
- Wind Speed

A target column (`Weather_Class`) is created using:

- Warm → Temperature ≥ 25°C
- Cool → Temperature < 25°C

---

## Technologies Used

- Python
- Pandas
- Scikit-learn

---

## Machine Learning Workflow

1. Load weather dataset
2. Data preprocessing
3. Create target variable
4. Label Encoding
5. Train-Test Split (80:20)
6. Feature Scaling using StandardScaler
7. Train SVM (RBF Kernel)
8. Evaluate the model

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Project Structure

```
Assignment-6/
│── Assignment-6.py
│── weather_data.csv
│── README.md
│── requirements.txt
│── .gitignore
```

---

## Results

The trained SVM model achieved approximately:

- Accuracy: **95.89%**
- Precision: **93.02%**
- Recall: **100%**
- F1 Score: **96.39%**

---

## Conclusion

The Support Vector Machine successfully classified weather conditions with high accuracy. Feature scaling using StandardScaler improved model performance because SVM is sensitive to feature magnitudes. The RBF kernel effectively separated the weather classes. While SVM provides excellent classification accuracy, it can become computationally expensive on larger datasets.

---

## Author

Ayush Dev