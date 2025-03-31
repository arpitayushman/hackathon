import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from src.data_preprocessing import load_data, clean_data, create_analysis_data
from src.feature_engineering import prepare_features_for_modeling
from src.model_training import train_amount_model, train_volume_model, optimize_hyperparameters
from src.prediction import predict_transactions

def setup_directories():
    """
    Create necessary directories
    """
    directories = ['data', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory setup complete")

def save_raw_data(data):
    """
    Save raw data to CSV file
    """
    # If the data is already a string with formatting, convert it to a dataframe
    if isinstance(data, str):
        # Split lines and parse as CSV
        lines = [line.strip() for line in data.strip().split('\n')]
        headers = lines[0].split('\t')
        
        rows = []
        for line in lines[1:]:
            values = line.split('\t')
            rows.append(values)
        
        df = pd.DataFrame(rows, columns=headers)
    else:
        df = data
        
    df.to_csv('data/raw_data.csv', index=False)
    print("Raw data saved to data/raw_data.csv")

def train_models():
    """
    Train the transaction amount and volume prediction models
    """
    print("Loading and preprocessing data...")
    raw_data = load_data('data/raw_data.csv')
    cleaned_data = clean_data(raw_data)
    amount_df, volume_df = create_analysis_data(cleaned_data)
    
    print("\nPreparing features for amount prediction model...")
    X_amount, y_amount = prepare_features_for_modeling(amount_df, 'avg_amount', prediction_type='amount')
    
    print("\nTraining transaction amount model...")
    amount_results, amount_model = train_amount_model(X_amount, y_amount)
    
    print("\nPreparing features for volume prediction model...")
    X_volume, y_volume = prepare_features_for_modeling(volume_df, 'txn_count', prediction_type='volume')
    
    print("\nTraining transaction volume model...")
    volume_results, volume_model = train_volume_model(X_volume, y_volume)
    
    # Optimize models if time permits
    # print("\nOptimizing hyperparameters for amount model...")
    # optimize_hyperparameters(X_amount, y_amount, model_type='amount')
    
    # print("\nOptimizing hyperparameters for volume model...")
    # optimize_hyperparameters(X_volume, y_volume, model_type='volume')
    
    print("\nModel training complete!")

def make_predictions(biller_category, future_date):
    """
    Make predictions for a specific biller category and future date
    """
    print(f"\nMaking predictions for {biller_category} on {future_date}...")
    results = predict_transactions(biller_category, future_date)
    
    if results is not None:
        print("\nPrediction Results:")
        print(results)
        
        # Save results to CSV
        results.to_csv(f'results/{biller_category}_{future_date.strftime("%Y%m%d")}_predictions.csv', index=False)
        print(f"Results saved to results/{biller_category}_{future_date.strftime('%Y%m%d')}_predictions.csv")

def main():
    """
    Main function to run the entire pipeline
    """
    parser = argparse.ArgumentParser(description='Transaction Amount and Volume Prediction')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--category', type=str, help='Biller category for prediction')
    parser.add_argument('--date', type=str, help='Future date for prediction (YYYY-MM-DD)')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of models')
    
    args = parser.parse_args()
    
    setup_directories()
    
    # Check if raw data exists, if not, prompt user to provide it
    if not os.path.exists('data/raw_data.csv'):
        print("Raw data not found. Please provide data or specify its location.")
        # In a real application, you might have a way to input data here
    
    if args.train or args.retrain:
        train_models()
    
    if args.predict:
        if not args.category or not args.date:
            print("Error: Both --category and --date are required for prediction.")
            return
        
        try:
            future_date = pd.to_datetime(args.date)
            make_predictions(args.category, future_date)
        except Exception as e:
            print(f"Error making predictions: {e}")
    
    if not args.train and not args.predict and not args.retrain:
        # Default: run everything
        train_models()
        
        # Make sample predictions
        # Get unique categories from data
        raw_data = load_data('data/raw_data.csv')
        cleaned_data = clean_data(raw_data) 
        categories = cleaned_data['blr_category'].unique()
        
        if len(categories) > 0:
            # Make prediction for the first category, 30 days in the future
            future_date = datetime.now() + timedelta(days=30)
            make_predictions(categories[0], future_date)

if __name__ == "__main__":
    main()