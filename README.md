# Immo Eliza - Real Estate Price Prediction ML Project

## Project Context

This project aims to predict a price for properties (houses/appartments) in the Belgian real estate market based on a range of features.

## Project Structure:

```
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

## Data

- **Model**: Trained on property listings in Belgium, focusing on houses and appartments. 
- **Target**: Listing price of each property.
- **Features**: Includes number of beedroms, number of facades, construction year, property subtype, zip_code, total living area, dummy features (garden/terrace/) etc.  

## Model Details

- **Tested Models**: Multiple Linear Regression, XGBoost
- **Chosen Model**: XGBoost was selected for its balance of performance and interpretability.

## Performance

The model's performance was evaluated using several metrics to assess its predictive accuracy:

    R² Score: The XGB model achieved an R² score of 0.7426, explaining approximately 74% of the variance in property prices.

    Mean Absolute Error (MAE) (€42,079.76): On average, predictions deviate from actual prices by €42,079.76. This error is relatively high and reflects areas where model precision can improve. 
   
    Median Absolute Error (€30,489.45): The median error, or the midpoint of all errors, is €30,489.45, slightly lower than the MAE. This can suggest that some predictions have more extreme errors, skewing the mean.

## Comparison of Actual vs. Predicted Property Prices (n=10)
Table 1: Comparison of Actual vs. Predicted Property Prices

```go
   Actual Price  Predicted Price  Absolute Error  Percentage Error
0      345000.0    324148.375000    20851.625000          6.043949
1      399000.0    432871.906250    33871.906250          8.489200
2      570000.0    455934.906250   114065.093750         20.011420
3      479000.0    428170.968750    50829.031250         10.611489
4      373673.0    392023.875000    18350.875000          4.910945
5      439000.0    506548.250000    67548.250000         15.386845
6      399000.0    502941.531250   103941.531250         26.050509
7      375000.0    331482.781250    43517.218750         11.604592
8      170000.0    137094.484375    32905.515625         19.356186
9      295000.0    215162.250000    79837.750000         27.063644
```
Percentage Errors: Smaller properties or lower-priced listings tend to have higher percentage errors, as seen with 30.93% for row 4 and 32.83% for row 10. Higher-priced properties may have relatively lower percentage errors but higher absolute errors.

## Usage Guide:

### Dependencies:

Install dependencies from `requirements.txt`. Main libraries: `pandas`, `scikit-learn`, `joblib`, `numpy`, `xgboost`.

### Training the Model:

Run `train.py` to train the model (under construction)

### Generating Predictions:

Use `predict.py` with new data in the same format as the training set to generate predictions (under construction)

## Limitations and Future Work:

- Potential enhancements, like addressing multicollinearity or further data enrichment (e.g., macroeconomic indicators).
- Explore more advanced models and feature engineering / importance techniques.
- Refining the feature set (e.g., adding location granularity, time trends)
- Handling outliers more robustly
- Experimenting with log transformation of target and numerical variables to mitigate skewed data.
