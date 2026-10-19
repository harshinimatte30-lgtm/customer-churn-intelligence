---
title: Customer Churn Intelligence
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Customer Churn Intelligence

An end-to-end machine learning project that estimates customer
churn probability using customer service and billing information.

## Features

- Interactive customer input form
- Churn probability estimation
- Low, medium and high risk categories
- Preprocessing and prediction in a saved ML pipeline

## Machine Learning

- Logistic Regression baseline
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Precision, recall, F1-score and confusion matrix evaluation
- Exploratory analysis of contract type and customer tenure

## Dataset

Telco Customer Churn dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Limitations

This is an educational portfolio project. Predictions are estimates,
not guarantees of customer behaviour. Risk thresholds are preliminary
and have not been validated for real business decisions.

## Technology

Python, pandas, scikit-learn, joblib and Gradio.
