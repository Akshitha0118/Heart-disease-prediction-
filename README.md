# End-to-End Clinical Heart Disease Prediction using Machine Learning Models
end-to-end Machine Learning project focused on Heart Disease Prediction using clinical patient data and advanced model evaluation techniques.

## 🔍 Project Objective:
 The goal of this project was to predict the presence of heart disease and perform a comparative analysis of multiple classification models to identify the best-performing algorithm based on robust evaluation metrics.

## 📊 Dataset & Preprocessing:
 I used the UCI Heart Disease Dataset and applied several preprocessing steps including:
 • Handling missing values using imputation
 • One-hot encoding categorical features
 • Feature scaling with StandardScaler

## 📊 Machine Learning Model Evaluation Dashboard

An interactive Streamlit-based dashboard for evaluating and comparing multiple classification models using key performance metrics such as Accuracy, Weighted F1-score, Confusion Matrix, ROC-AUC, and Precision–Recall.

This project focuses on model evaluation, error analysis, and performance interpretation, making it highly valuable for Data Scientist and Machine Learning Engineer portfolios.

## 🔍 Project Overview

Choosing the right machine learning model is not just about accuracy—it requires understanding how models behave, where they fail, and how they trade off precision and recall.

### This dashboard provides:

Side-by-side comparison of multiple ML classifiers

Confusion matrix visualizations

Automatic best-model identification

Clear metric-based decision support

## 🧠 Models Evaluated

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

XGBoost

Naive Bayes

Gradient Boosting

AdaBoost

Extra Trees

## 🛠️ Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

## ✨ Key Features

📊 Interactive Streamlit dashboard

📋 Model comparison table (Accuracy & Weighted F1-score)

🧩 Confusion matrix visualization for multiple models

🏆 Automatic best-performing model detection

🎨 Clean UI with custom CSS styling

📱 Responsive wide-layout design

## 📊 Model Performance Summary

The dashboard displays a comparison table containing:

Accuracy

Weighted F1-score

These metrics help identify models that balance class-level performance rather than favoring dominant classes.

## 🧩 Confusion Matrix Analysis

Confusion matrices are visualized for selected models to analyze:

True Positives

True Negatives

False Positives

False Negatives

This helps in understanding:

Which classes are misclassified

Model bias toward specific classes

Error distribution patterns

## 📈 ROC-AUC Explained

ROC-AUC (Receiver Operating Characteristic – Area Under Curve) measures how well a model can distinguish between classes across different probability thresholds.

### Why ROC-AUC Matters:

Threshold-independent metric

Measures overall class separability

Higher AUC = better model discrimination

## Interpretation:

AUC = 0.5 → No discrimination (random guessing)

AUC = 1.0 → Perfect classification

ROC-AUC is especially useful when:

Class imbalance exists

Probability-based predictions are required

## 📉 Precision–Recall Explained

Precision–Recall Curve focuses on the performance of the positive class, making it critical for imbalanced datasets.

## Key Metrics:

Precision → How many predicted positives are actually positive

Recall → How many actual positives were correctly identified

## Why Precision–Recall is Important:

More informative than ROC-AUC for rare classes

Highlights trade-off between false positives and false negatives

📌 In real-world use cases (fraud detection, churn prediction, medical diagnosis), Precision–Recall often matters more than accuracy.

## 🏆 Best Model Selection

The dashboard automatically identifies the best-performing model based on:

Highest Accuracy

Strong Weighted F1-score

This helps users quickly select a production-ready model while still analyzing trade-offs visually.

## ▶️ How It Works

Model metrics are stored in a structured dataset.

Accuracy & F1-scores are displayed in a comparison table.

Confusion matrices are plotted for selected models.

Best model is highlighted dynamically.

Users interpret results using multiple evaluation perspectives.

## ▶️ Usage

Run the dashboard locally using:

streamlit run app.py

## 📌 Output

The dashboard provides:

Model comparison table

Confusion matrix visualizations

Best model summary

Clear performance interpretation
