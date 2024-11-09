
# Immo Eliza - Real Estate Price Prediction

> **A Machine Learning Project to Predict Property Prices in the Belgian Market**

This project aims to predict prices for residential properties (houses and apartments) in Belgium based on a range of property features. Separate models were trained for apartments and houses to improve prediction accuracy based on property type.

---

## 📂 Project Structure

```plaintext
ml_immoweb/
├── Data/
│   ├── Comparison/
│   ├── Preprocessed/
│   ├── Raw/
├── Notebooks/
├── scripts/
│   ├── train.py
│   ├── predict.py
├── Trained_Models/
├── requirements.txt
├── README.md
```

## 📊 Data Overview

- **Scope**: Belgian residential property listings, focusing on houses and apartments.
- **Target Variable**: Listing price.
- **Features**: A variety of property characteristics, including the number of bedrooms, number of facades, construction year, zip code, total living area, dummy variables (garden, terrace, etc.).

---

## 🚀 Model Details

- **Explored Models**: Multiple Linear Regression, Random Forest, XGBoost.
- **Final Model**: XGBoost was selected for its balance of performance and interpretability, providing a reliable prediction model for this real estate data.

---

## 📈 Performance Metrics

The XGBoost model’s performance was assessed using several metrics:

- **R² Score**: 0.7426 – Explains approximately 74% of the variance in property prices.
- **Mean Absolute Error (MAE)**: €42,079.76 – Average deviation from actual prices.
- **Median Absolute Error**: €30,489.45 – Indicates some extreme errors that skew the mean.

### Sample Comparison of Actual vs. Predicted Prices (n=10)

| Actual Price | Predicted Price | Absolute Error | Percentage Error |
|--------------|-----------------|----------------|------------------|
| 345,000      | 324,148         | 20,852        | 6.04%           |
| 399,000      | 432,872         | 33,872        | 8.49%           |
| 570,000      | 455,935         | 114,065       | 20.01%          |
| 479,000      | 428,171         | 50,829        | 10.61%          |
| 373,673      | 392,024         | 18,351        | 4.91%           |
| 439,000      | 506,548         | 67,548        | 15.39%          |
| 399,000      | 502,942         | 103,942       | 26.05%          |
| 375,000      | 331,483         | 43,517        | 11.60%          |
| 170,000      | 137,094         | 32,906        | 19.36%          |
| 295,000      | 215,162         | 79,838        | 27.06%          |

- **Insights**: Higher-priced properties have larger absolute errors but lower percentage errors, while smaller or lower-priced properties tend to have higher percentage errors.

---

## ⚙️ Usage Guide

### Installation

To install dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Training the Model

Use `train.py` to train the model:

```bash
python scripts/train.py
```

### Generating Predictions

Generate predictions using `predict.py` with new data in the same format as the training set:

```bash
python scripts/predict.py
```

---

## 📂 Project Background & Timeline

This is the third phase of a four-phase project to create a complete ML pipeline for predicting residential property prices. T
his project phase took one week to complete in October 2024. 

The project was completed as part of my 7-month AI & Data Science bootcamp at BeCode in Ghent, Belgium.

## 🔍 Limitations & Future Work

- **Data Enhancements**: Add macroeconomic indicators, time trends, and additional location granularity.
- **Feature Engineering**: Address multicollinearity and further optimize feature importance techniques.
- **Robust Outlier Handling**: Experiment with transformations to mitigate skewed data.
- **Advanced Modeling**: Explore alternative models and further tuning options to improve MAE and R² scores.

---

## 📊 Performance Summary
The XGBoost model was chosen for its efficient handling of complex patterns in data. 


## 📫 Contact

For questions or further information, please reach out to [Urson Callens](https://www.github.com/ursonc).

---
