import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from .feature_engineering import create_time_features, create_historical_features, encode_categorical_features
from .data_preprocessing import load_data, clean_data, create_analysis_data

def load_models():
    """
    Load the trained prediction models
    """
    amount_model_path = 'models/amount_model.joblib'
    volume_model_path = 'models/volume_model.joblib'
    
    if not os.path.exists(amount_model_path) or not os.path.exists(volume_model_path):
        print("Models not found. Please train models first.")
        return None, None
    
    try:
        amount_model = joblib.load(amount_model_path)
        volume_model = joblib.load(volume_model_path)
        print("Models loaded successfully")
        return amount_model, volume_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None
    
def ensure_feature_compatibility(pred_data, model_type='amount'):
    """
    Ensures that the prediction data has the exact features needed by the model
    """
    try:
        # Load the required features
        model_features_file = f'models/{model_type}_model_features.joblib'
        if os.path.exists(model_features_file):
            required_features = joblib.load(model_features_file)
        else:
            # If no feature file exists, load the model and get its features
            model_file = f'models/{model_type}_model.joblib'
            model = joblib.load(model_file)
            if hasattr(model, 'feature_names_in_'):
                required_features = model.feature_names_in_
            else:
                print(f"Cannot determine required features for {model_type} model.")
                return pred_data
        
        # Remove metadata columns that shouldn't be used for prediction
        if 'original_category' in pred_data.columns:
            pred_data = pred_data.drop('original_category', axis=1)
        if 'original_date' in pred_data.columns:
            pred_data = pred_data.drop('original_date', axis=1)
        
        # Create a new DataFrame with only the required features
        compatible_data = pd.DataFrame()
        
        # Add required features
        for feature in required_features:
            if feature in pred_data.columns:
                compatible_data[feature] = pred_data[feature]
            else:
                # If feature is missing, add it with zeros
                print(f"Adding missing feature: {feature}")
                compatible_data[feature] = 0
        
        return compatible_data
    
    except Exception as e:
        print(f"Error ensuring feature compatibility: {e}")
        return pred_data

def prepare_prediction_data(df, biller_category, future_date):
    """
    Prepare data for making predictions
    """
    # Convert future_date to datetime if it's not already
    if isinstance(future_date, str):
        future_date = pd.to_datetime(future_date)
    
    # Create a list of dates from the most recent date in the data to the future date
    latest_date = df['txn_date'].max()
    
    # If future date is before the latest date in our data, use the latest date
    if future_date <= latest_date:
        print(f"Warning: Future date {future_date} is before or equal to the latest date in the data {latest_date}.")
        future_date = latest_date + timedelta(days=1)
    
    # Create date range
    prediction_dates = pd.date_range(start=latest_date + timedelta(days=1), end=future_date)
    
    # Create prediction dataframe
    prediction_df = pd.DataFrame({
        'txn_date': prediction_dates,
        'blr_category': biller_category,
        # Add placeholder values for required fields
        'avg_amount': np.nan,
        'total_amount': np.nan,
        'txn_count': np.nan
    })
    
    # Concatenate with original data (filtered for the specific category)
    category_df = df[df['blr_category'] == biller_category].copy()
    
    # If no data for this category, use the most similar category or all data
    if len(category_df) == 0:
        print(f"Warning: No historical data found for category '{biller_category}'. Using average data.")
        # Use average values across all categories
        category_df = df.copy()
    
    combined_df = pd.concat([category_df, prediction_df], ignore_index=True)
    combined_df = combined_df.sort_values('txn_date')
    
    # Ensure we have baseline values for the new prediction rows
    # Fill with the average for the category
    for col in ['avg_amount', 'total_amount', 'txn_count']:
        avg_value = category_df[col].mean()
        combined_df[col] = combined_df[col].fillna(avg_value)
    
    # Create features for prediction
    combined_df = create_time_features(combined_df)
    combined_df = create_historical_features(combined_df)
    
    # Extract only the prediction rows (future dates)
    prediction_rows = combined_df[combined_df['txn_date'] >= prediction_dates[0]].copy()
    
    # Encode categorical features (with training mode disabled)
    encoded_df = encode_categorical_features(prediction_rows, training_mode=False)
    
    # Ensure all required features are present
    # Don't remove total_amount and txn_count as they're needed for prediction
    
    print(f"Data prepared for prediction with {len(prediction_dates)} future dates")
    return encoded_df, prediction_rows

def predict_transactions(biller_category, future_date):
    """
    Predict transaction amount and volume for a specific biller category and future date
    """
    # Load models
    amount_model, volume_model = load_models()
    if amount_model is None or volume_model is None:
        return None
    
    # Load and preprocess data
    raw_data = load_data('data/raw_data.csv')
    cleaned_data = clean_data(raw_data)
    amount_df, volume_df = create_analysis_data(cleaned_data)
    
    # Prepare prediction data
    pred_data, pred_rows = prepare_prediction_data(amount_df, biller_category, future_date)
    
    try:
        # Ensure feature compatibility for both models
        amount_features = ensure_feature_compatibility(pred_data, 'amount')
        volume_features = ensure_feature_compatibility(pred_data, 'volume')
        
        # Double-check that no metadata columns remain
        metadata_cols = ['original_category', 'original_date']
        for col in metadata_cols:
            if col in amount_features.columns:
                amount_features = amount_features.drop(col, axis=1)
            if col in volume_features.columns:
                volume_features = volume_features.drop(col, axis=1)
        
        # Make predictions
        amount_pred = amount_model.predict(amount_features)
        volume_pred = volume_model.predict(volume_features)
        
        # Create results dataframe
        results = pd.DataFrame({
            'date': pred_rows['txn_date'],
            'biller_category': biller_category,
            'predicted_avg_amount': amount_pred,
            'predicted_txn_count': volume_pred,
            'predicted_total_amount': amount_pred * volume_pred
        })
        
        # Visualize predictions
        visualize_predictions(results, biller_category)
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nTrying alternative approach...")
        
        # If we have model feature information, try to display it
        try:
            if hasattr(amount_model, 'feature_names_in_'):
                print("\nAmount model expects these features:")
                print(amount_model.feature_names_in_)
                
                # Show which features are missing from prediction data
                missing = set(amount_model.feature_names_in_) - set(pred_data.columns)
                print(f"\nMissing features: {missing}")
            
            # Show what features are available in the prediction data
            print("\nAvailable features in prediction data:")
            print(pred_data.columns.tolist())
            
            # Suggest a solution
            print("\nRecommendation: Retrain your model with the 'total_amount' and 'txn_count' features included.")
            print("Or update the prediction code to include these features.")
        except:
            pass
        
        return None

def visualize_predictions(predictions, biller_category):
    """
    Create visualizations of the predictions
    """
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Transaction Amount Prediction
    plt.subplot(2, 1, 1)
    plt.plot(predictions['date'], predictions['predicted_avg_amount'], marker='o', linestyle='-', color='#1f77b4')
    plt.title(f'Predicted Average Transaction Amount for {biller_category}')
    plt.xlabel('Date')
    plt.ylabel('Average Amount')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 2: Transaction Volume Prediction
    plt.subplot(2, 1, 2)
    plt.plot(predictions['date'], predictions['predicted_txn_count'], marker='o', linestyle='-', color='#ff7f0e')
    plt.title(f'Predicted Transaction Volume for {biller_category}')
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{biller_category}_predictions.png')
    plt.close()
    
    # Create a second figure for total amount
    plt.figure(figsize=(10, 6))
    plt.plot(predictions['date'], predictions['predicted_total_amount'], marker='o', linestyle='-', color='#2ca02c')
    plt.title(f'Predicted Total Transaction Amount for {biller_category}')
    plt.xlabel('Date')
    plt.ylabel('Total Amount')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the second figure
    plt.savefig(f'results/{biller_category}_total_amount.png')
    plt.close()
    
    print(f"Predictions visualized and saved to results/{biller_category}_predictions.png")