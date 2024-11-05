import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define features
num_features = ["total_area_sqm", "nbr_frontages", "nbr_bedrooms"]
cat_features = ["state_building", "province", "region", "subproperty_type"]
dummy_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace", 
                  "fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]
zip_feature = ["zip_code"]

def clean_data(df):
    # Cleaning steps (filtering frontages, bedrooms, regions, and outliers)
    df = df[(df['nbr_frontages'] >= 1) & (df['nbr_frontages'] <= 6) &
            (df['nbr_bedrooms'] >= 1) & (df['nbr_bedrooms'] <= 8)]
    df = df[df['region'] != 'MISSING']
    for prop_type in df['subproperty_type'].unique():
        q1, q3 = df[df['subproperty_type'] == prop_type]['price'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    return df

# Load and clean data
df = pd.read_csv('properties_raw.csv')
df = clean_data(df)

# Split data into features and target
X = df[num_features + dummy_features + cat_features + zip_feature]
y = df['price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=535)

# Define preprocessing pipeline with Target Encoding for zip_code and median imputation for numerical features
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])

# Apply One-Hot Encoding for other categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('dummy', 'passthrough', dummy_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ]
)

# Preprocess data (fit on training data without zip_code)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply Target Encoding for zip_code separately
zip_encoder = ce.TargetEncoder(cols=['zip_code'])
X_train['zip_code'] = zip_encoder.fit_transform(X_train['zip_code'], y_train)
X_test['zip_code'] = zip_encoder.transform(X_test['zip_code'])

# Concatenate the processed columns and zip_code encoding
X_train_final = pd.concat([pd.DataFrame(X_train_processed), X_train['zip_code'].reset_index(drop=True)], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_processed), X_test['zip_code'].reset_index(drop=True)], axis=1)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=535)
model.fit(X_train_final, y_train)

# Save the model, preprocessor, and zip encoder together as a dictionary
joblib.dump({"model": model, "preprocessor": preprocessor, "zip_encoder": zip_encoder}, "Trained_Models/trained_xgb_model.joblib")

# Evaluate model performance
y_pred_test = model.predict(X_test_final)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

print(f"R² Test: {r2_test:.4f}")
print(f"MAE Test: {mae_test:.2f}")
print(f"RMSE Test: {rmse_test:.2f}")
