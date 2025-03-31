import argparse
import joblib
import pandas as pd
from utils import prepare_prediction_data

def predict_future_amount(model_path, biller_category, future_date):
    """
    Predict transaction amount for a specific biller category on a future date
    
    Parameters:
    model_path (str): Path to trained model
    biller_category (str): Biller category to predict for
    future_date (str or datetime): Future date (YYYY-MM-DD) or month (YYYY-MM)
    
    Returns:
    float: Predicted transaction amount
    """
    # Load model components
    model_components = joblib.load(model_path)
    model = model_components['model']
    encoders = model_components['encoders']
    scaler = model_components['scaler']
    feature_cols = model_components['feature_cols']
    
    # Parse the date input
    if isinstance(future_date, str):
        if len(future_date.split('-')) == 2:  # Format: YYYY-MM
            # Add day as the first of month if only month is specified
            future_date = pd.to_datetime(future_date + '-01')
        else:
            future_date = pd.to_datetime(future_date)
    
    # Create a sample dataframe with minimal required fields
    sample = pd.DataFrame({
        'blr_category': [biller_category],
        'crtn_ts': [future_date]
    })
    
    # Fill other required fields with defaults
    default_columns = ['txn_type', 'mti', 'response_code', 'payment_channel', 
                      'cou_id', 'bou_id', 'bou_status', 'payment_mode', 'on_us']
    for col in default_columns:
        # Use most common value from training if available
        if col in encoders:
            sample[col] = encoders[col].classes_[0]
        else:
            sample[col] = 'unknown'
    
    # Process sample through the same pipeline
    X, _ = prepare_prediction_data(sample, encoders, feature_cols, scaler)
    
    # Make prediction
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    return prediction
def main():
    parser = argparse.ArgumentParser(description='Forecast transaction amounts for future dates')
    parser.add_argument('--model', type=str, default='models/transaction_predictor.mdl', 
                      help='Path to the trained model file')
    parser.add_argument('--category', type=str, required=True, 
                      help='Biller category to forecast for')
    parser.add_argument('--date', type=str, required=True, 
                      help='Future date (YYYY-MM-DD) or month (YYYY-MM)')
    args = parser.parse_args()
    
    # Make prediction
    predicted_amount = predict_future_amount(args.model, args.category, args.date)
    
    print(f"\nForecast for {args.category} on {args.date}:")
    print(f"Predicted transaction amount: ₹{predicted_amount:.2f}")

if __name__ == "__main__":
    main()