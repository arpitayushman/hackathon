import os
import pandas as pd
import numpy as np
import joblib
import argparse
from utils import load_data, preprocess_data, engineer_features

def load_model(model_path):
    """Load the trained model and its components"""
    try:
        model_components = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model_components
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_prediction_data(df, encoders, feature_cols):
    """Prepare data for prediction using the same preprocessing steps as training"""
    # Preprocess the data
    df_processed, _ = preprocess_data(df)
    
    # Use existing encoders from training
    for col, encoder in encoders.items():
        if col in df_processed.columns:
            # Transform using existing encoder, handle unseen categories
            try:
                df_processed[f'{col}_encoded'] = encoder.transform(df_processed[col])
            except ValueError:
                # For unseen categories, use a default value
                print(f"Warning: Unseen categories in {col}. Using default encoding.")
                df_processed[f'{col}_encoded'] = 0
    
    # Engineer features
    df_featured = engineer_features(df_processed, encoders)
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in df_featured.columns:
            print(f"Warning: Feature {col} not found. Adding with zeros.")
            df_featured[col] = 0
    
    # Select only the features used during training
    X = df_featured[feature_cols]
    
    return X, df_featured

def make_predictions(model, X, scaler):
    """Make predictions using the trained model"""
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

def save_predictions(df, predictions, output_file):
    """Save predictions to a CSV file"""
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Add predictions
    result_df['predicted_amount'] = predictions
    
    # Add error if actual values are available
    if 'txn_amount' in result_df.columns:
        result_df['prediction_error'] = result_df['predicted_amount'] - result_df['txn_amount']
        result_df['error_percentage'] = (result_df['prediction_error'] / result_df['txn_amount']) * 100
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Predict transaction amounts using trained model')
    parser.add_argument('--model', type=str, default='models/transaction_classifier.mdl', help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='data/sample.csv', help='Path to the CSV data file for prediction')
    parser.add_argument('--output', type=str, default='output/predictions.csv', help='Path to save prediction results')
    args = parser.parse_args()
    
    # Load the model
    model_components = load_model(args.model)
    if model_components is None:
        return
    
    # Extract model components
    model = model_components['model']
    encoders = model_components['encoders']
    scaler = model_components['scaler']
    feature_cols = model_components['feature_cols']
    
    # Load data
    df = load_data(args.data)
    if df is None:
        return
    
    # Prepare data for prediction
    X, df_featured = prepare_prediction_data(df, encoders, feature_cols)
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, X, scaler)
    
    # Save predictions
    result_df = save_predictions(df, predictions, args.output)
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Mean Predicted Amount: {predictions.mean():.2f}")
    print(f"Min Predicted Amount: {predictions.min():.2f}")
    print(f"Max Predicted Amount: {predictions.max():.2f}")
    
    if 'txn_amount' in df.columns:
        errors = predictions - df['txn_amount'].values
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        
        print("\nError Metrics on New Data:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")

if __name__ == "__main__":
    main()