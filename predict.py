import pandas as pd
import joblib

# Load the saved model directly from the updated path
model = joblib.load("Trained_Models/appartments_xgb_model.joblib")  # Adjusted path

def predict_price(new_data):
    """
    Predict the price of a new property based on the trained model.
    """
    # Assuming `new_data` is already preprocessed in the same format as the training data
    predicted_price = model.predict(new_data)
    return predicted_price

# Example usage with preprocessed new property data
# Ensure `new_property` matches the format and preprocessing of your training data
new_property = pd.DataFrame({
    'total_area_sqm': [120],
    'nbr_frontages': [2],
    'nbr_bedrooms': [3],
    'zip_code': [1000],        
    'state_building': [2],     
    'province': [1],           
    'region': [1],            
    'subproperty_type': [0],   
    'fl_garden': [1],
    'fl_furnished': [0],
    'fl_open_fire': [0],
    'fl_terrace': [1],
    'fl_swimming_pool': [0],
    'fl_floodzone': [0],
    'fl_double_glazing': [1]
})

# Predict and print the result
predicted_price = predict_price(new_property)
print(f"Predicted Price: {predicted_price[0]:.2f}")
