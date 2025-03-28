import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import custom modules
from src.data_preprocessing import load_data, clean_data, create_analysis_data
from src.feature_engineering import prepare_features_for_modeling
from src.model_training import (
    train_amount_model, 
    train_volume_model, 
    optimize_hyperparameters
)
from src.prediction import predict_transactions

def setup_directories():
    """
    Create necessary directories for the project
    """
    directories = ['data', 'models', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory setup complete")

def save_training_logs(results, prediction_type):
    """
    Save model training results to a log file
    """
    log_file = f'logs/{prediction_type}_model_training_log.txt'
    
    with open(log_file, 'w') as f:
        f.write(f"Model Training Results for {prediction_type.capitalize()} Prediction\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, model_info in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("Best Parameters:\n")
            for param, value in model_info['best_params'].items():
                f.write(f"  {param}: {value}\n")
            
            f.write("\nPerformance Metrics:\n")
            for metric, value in model_info['performance'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n" + "-" * 40 + "\n\n")

def train_models(retrain=False):
    """
    Comprehensive model training pipeline
    """
    print("Starting model training pipeline...")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    raw_data = load_data('data/raw_data.csv')
    
    if raw_data is None:
        print("Error: Unable to load raw data. Please check the data file.")
        return
    
    cleaned_data = clean_data(raw_data)
    amount_df, volume_df = create_analysis_data(cleaned_data)
    
    # Prepare features for modeling
    print("\nPreparing features for amount prediction model...")
    X_amount, y_amount = prepare_features_for_modeling(amount_df, 'avg_amount', prediction_type='amount')
    
    print("\nPreparing features for volume prediction model...")
    X_volume, y_volume = prepare_features_for_modeling(volume_df, 'txn_count', prediction_type='volume')
    
    # Train models
    print("\nTraining transaction amount prediction model...")
    amount_results, amount_model = train_amount_model(X_amount, y_amount)
    save_training_logs(amount_results, 'amount')
    
    print("\nTraining transaction volume prediction model...")
    volume_results, volume_model = train_volume_model(X_volume, y_volume)
    save_training_logs(volume_results, 'volume')
    
    # Optional: Hyperparameter optimization (uncomment if needed)
    # print("\nOptimizing amount model hyperparameters...")
    # optimize_hyperparameters(X_amount, y_amount, model_type='amount')
    
    # print("\nOptimizing volume model hyperparameters...")
    # optimize_hyperparameters(X_volume, y_volume, model_type='volume')
    
    print("\nModel training complete!")

def make_predictions(biller_category, future_date):
    """
    Make predictions for a specific biller category and future date
    """
    print(f"\nMaking predictions for {biller_category} on {future_date}...")
    
    try:
        results = predict_transactions(biller_category, future_date)
        
        if results is not None:
            print("\nPrediction Results:")
            print(results)
            
            # Save results to CSV
            results.to_csv(
                f'results/{biller_category}_{future_date.strftime("%Y%m%d")}_predictions.csv', 
                index=False
            )
            print(f"Results saved to results/{biller_category}_{future_date.strftime('%Y%m%d')}_predictions.csv")
    except Exception as e:
        print(f"Error making predictions: {e}")

def main():
    """
    Main function to control the entire pipeline
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Transaction Prediction Pipeline')
    
    # Add arguments
    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Train the prediction models'
    )
    parser.add_argument(
        '--predict', 
        action='store_true', 
        help='Make predictions'
    )
    parser.add_argument(
        '--category', 
        type=str, 
        help='Biller category for prediction'
    )
    parser.add_argument(
        '--date', 
        type=str, 
        help='Future date for prediction (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--retrain', 
        action='store_true', 
        help='Force retraining of models'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup project directories
    setup_directories()
    
    # Check if raw data exists
    if not os.path.exists('data/raw_data.csv'):
        print("Error: Raw data file not found at data/raw_data.csv")
        sys.exit(1)
    
    # Training logic
    if args.train or args.retrain:
        train_models()
    
    # Prediction logic
    if args.predict:
        # Validate prediction arguments
        if not args.category or not args.date:
            print("Error: Both --category and --date are required for prediction.")
            sys.exit(1)
        
        try:
            future_date = pd.to_datetime(args.date)
            make_predictions(args.category, future_date)
        except Exception as e:
            print(f"Prediction error: {e}")
    
    # Default behavior if no specific action is specified
    if not args.train and not args.predict and not args.retrain:
        # Default: train models and make a sample prediction
        train_models()
        
        # Load data to get categories
        raw_data = load_data('data/raw_data.csv')
        cleaned_data = clean_data(raw_data)
        categories = cleaned_data['blr_category'].unique()
        
        if len(categories) > 0:
            # Make prediction for the first category, 30 days in the future
            future_date = datetime.now() + timedelta(days=30)
            make_predictions(categories[0], future_date)

if __name__ == "__main__":
    main()