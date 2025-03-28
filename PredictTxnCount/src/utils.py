import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """Load the CSV data file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert timestamp to datetime
    df_processed['crtn_ts'] = pd.to_datetime(df_processed['crtn_ts'])
    
    # Extract features from datetime
    df_processed['month'] = df_processed['crtn_ts'].dt.month
    df_processed['day'] = df_processed['crtn_ts'].dt.day
    df_processed['hour'] = df_processed['crtn_ts'].dt.hour
    df_processed['dayofweek'] = df_processed['crtn_ts'].dt.dayofweek
    
    # Handle categorical variables
    categorical_cols = ['txn_type', 'mti', 'blr_category', 'response_code', 
                        'payment_channel', 'cou_id', 'bou_id', 'bou_status',
                        'payment_mode', 'on_us']
    
    # Create label encoders for each categorical variable
    encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            encoders[col] = le
    
    # Handle missing values
    df_processed = df_processed.fillna(0)
    
    return df_processed, encoders

def prepare_prediction_data(sample, encoders, feature_cols, scaler):
    """
    Prepare incoming data for prediction by encoding categorical variables and scaling numerical features.
    
    Parameters:
        sample (dict): Dictionary containing input feature values.
        encoders (dict): Pre-fitted encoders for categorical features.
        feature_cols (list): List of feature column names used in training.
        scaler (StandardScaler): Pre-fitted scaler used for numerical features.
    
    Returns:
        np.ndarray: Transformed feature array ready for model prediction.
    """
    
    # Convert sample dictionary to DataFrame
    # sample_df = pd.DataFrame([sample])
    if isinstance(sample, np.ndarray):
        sample = sample.reshape(1, -1)  # Ensure it's 2D

    elif isinstance(sample, list) and isinstance(sample[0], list):
        sample = sample[0]  # Flatten nested list

    sample_df = pd.DataFrame([sample], columns=feature_cols)

    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in sample_df:
            if sample_df[col].iloc[0] in encoder.classes_:
                sample_df[col] = encoder.transform(sample_df[col])
            else:
                sample_df[col] = encoder.transform(["unknown"])  # Assigning 'unknown' category if unseen
    
    # Ensure all expected features exist
    for col in feature_cols:
        if col not in sample_df:
            sample_df[col] = 0  # Default value for missing features
    
    # Reorder columns to match training feature order
    sample_df = sample_df[feature_cols]
    
    # Scale numerical features
    sample_scaled = scaler.transform(sample_df)
    
    return sample_scaled

def engineer_features(df, encoders):
    """Enhanced feature engineering function"""
    df_featured = df.copy()
    
    # Original features
    biller_avg_txn = df_featured.groupby('blr_category')['txn_amount'].transform('mean')
    df_featured['biller_avg_txn'] = biller_avg_txn
    channel_counts = df_featured.groupby('payment_channel')['txn_amount'].transform('count')
    df_featured['channel_txn_count'] = channel_counts
    
    # New features
    
    # Success/failure indicators
    df_featured['is_success'] = (df_featured['response_code'] == '000').astype(int)
    
    # Transaction type indicators
    df_featured['is_cross_bank'] = (df_featured['on_us'] == 'N').astype(int)
    
    # Category statistics
    biller_stats = df_featured.groupby('blr_category')['txn_amount'].agg(['median', 'std']).reset_index()
    biller_stats.columns = ['blr_category', 'biller_median_txn', 'biller_std_txn']
    df_featured = pd.merge(df_featured, biller_stats, on='blr_category', how='left')
    
    # Payment mode features
    df_featured['is_card'] = ((df_featured['payment_mode'] == 'Credit_Card') | 
                            (df_featured['payment_mode'] == 'Debit_Card')).astype(int)
    df_featured['is_cash'] = (df_featured['payment_mode'] == 'Cash').astype(int)
    df_featured['is_digital'] = ((df_featured['payment_mode'] == 'Net_Banking') | 
                               (df_featured['payment_mode'] == 'Wallet') | 
                               (df_featured['payment_mode'] == 'IMPS')).astype(int)
    
    # Z-score within category
    df_featured['txn_category_zscore'] = (df_featured['txn_amount'] - df_featured['biller_avg_txn']) / df_featured['biller_std_txn'].replace(0, 1)
    
    # Fill missing values
    df_featured = df_featured.fillna(0)
    
    return df_featured

def prepare_modeling_data(df, feature_cols=None):
    """Prepare features and target for modeling with enhanced feature set"""
    # Define enhanced feature set
    if feature_cols is None:
        feature_cols = [
            # Original features
            'payment_channel_encoded', 'payment_mode_encoded', 'response_code_encoded',
            'bou_status_encoded', 'on_us_encoded', 'blr_category_encoded',
            'month', 'day', 'hour', 'dayofweek',
            'biller_avg_txn', 'channel_txn_count',
            
            # New features
            'is_success', 'is_cross_bank', 
            'biller_median_txn', 'biller_std_txn',
            'is_card', 'is_cash', 'is_digital',
            'txn_category_zscore'
        ]
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found. Adding with zeros.")
            df[col] = 0
    
    # Select features and target
    X = df[feature_cols]
    y = df['txn_amount'] if 'txn_amount' in df.columns else None
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols

def create_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)