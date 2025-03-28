import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

def create_time_features(df):
    """
    Create additional time-based features for the dataset
    """
    if df is None or df.empty:
        print("No data for feature engineering")
        return None
    
    # Make a copy of the dataframe
    feature_df = df.copy()
    
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(feature_df['txn_date']):
        feature_df['txn_date'] = pd.to_datetime(feature_df['txn_date'])
    
    # Extract detailed time features
    feature_df['year'] = feature_df['txn_date'].dt.year
    feature_df['month'] = feature_df['txn_date'].dt.month
    feature_df['day'] = feature_df['txn_date'].dt.day
    feature_df['day_of_week'] = feature_df['txn_date'].dt.dayofweek
    feature_df['is_weekend'] = feature_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    feature_df['quarter'] = feature_df['txn_date'].dt.quarter
    feature_df['is_month_start'] = feature_df['txn_date'].dt.is_month_start.astype(int)
    feature_df['is_month_end'] = feature_df['txn_date'].dt.is_month_end.astype(int)
    
    # Create month-end marker (for end-of-month patterns)
    feature_df['days_to_month_end'] = feature_df['txn_date'].dt.days_in_month - feature_df['txn_date'].dt.day
    
    # Apply cyclical encoding for month and day of week to capture seasonality
    feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
    feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
    feature_df['day_of_week_sin'] = np.sin(2 * np.pi * feature_df['day_of_week'] / 7)
    feature_df['day_of_week_cos'] = np.cos(2 * np.pi * feature_df['day_of_week'] / 7)
    
    return feature_df

def create_historical_features(df, window_sizes=[7, 14, 30]):
    """
    Create historical aggregation features (moving averages, etc.)
    """
    if df is None or df.empty:
        print("No data for creating historical features")
        return None
    
    # Make a copy of the dataframe
    hist_df = df.copy()
    
    # Ensure data is sorted by date
    hist_df = hist_df.sort_values(['blr_category', 'txn_date'])
    
    # Create historical features for each category separately
    result_dfs = []
    
    for category, category_df in hist_df.groupby('blr_category'):
        category_df = category_df.sort_values('txn_date')
        
        # Create moving averages for transaction amount and count
        for window in window_sizes:
            # Moving average for transaction amount
            category_df[f'avg_amount_last_{window}d'] = category_df['avg_amount'].rolling(window=window, min_periods=1).mean()
            category_df[f'total_amount_last_{window}d'] = category_df['total_amount'].rolling(window=window, min_periods=1).mean()
            category_df[f'txn_count_last_{window}d'] = category_df['txn_count'].rolling(window=window, min_periods=1).mean()
            
            # Trend indicators (slope over the window)
            category_df[f'amount_trend_{window}d'] = category_df['avg_amount'].diff(periods=window) / window
            category_df[f'volume_trend_{window}d'] = category_df['txn_count'].diff(periods=window) / window
        
        # Create lag features
        for lag in [1, 3, 7, 14]:
            category_df[f'avg_amount_lag_{lag}'] = category_df['avg_amount'].shift(lag)
            category_df[f'total_amount_lag_{lag}'] = category_df['total_amount'].shift(lag)
            category_df[f'txn_count_lag_{lag}'] = category_df['txn_count'].shift(lag)
        
        # Fill NaN values created by lags and rolling windows with appropriate values
        for col in category_df.columns:
            if '_lag_' in col or '_last_' in col or '_trend_' in col:
                # Group by biller category to calculate appropriate fill value
                fill_value = category_df[col].mean()
                category_df[col] = category_df[col].fillna(fill_value)
        
        result_dfs.append(category_df)
    
    # Combine all category dataframes
    final_df = pd.concat(result_dfs)
    
    return final_df

def encode_categorical_features(df, training_mode=False):
    """
    Encode categorical variables for machine learning models with consistency
    between training and prediction
    """
    if df is None or df.empty:
        print("No data for categorical encoding")
        return None
    
    # Make a copy of the dataframe
    encoded_df = df.copy()
    
    # One-hot encode biller category with a consistent approach
    if training_mode:
        # When training, create and save the encoding mapping
        encoded_cats = pd.get_dummies(encoded_df['blr_category'], prefix='cat')
        # Store the column names for future reference
        cat_columns = encoded_cats.columns.tolist()
        joblib.dump(cat_columns, 'models/category_columns.joblib')
        # Join with the main dataframe
        encoded_df = pd.concat([encoded_df.drop('blr_category', axis=1), encoded_cats], axis=1)
    else:
        # When predicting, use the same columns as during training
        try:
            cat_columns = joblib.load('models/category_columns.joblib')
            # Create one-hot encoding
            encoded_cats = pd.get_dummies(encoded_df['blr_category'], prefix='cat')
            # Ensure all training columns exist (add missing with zeros)
            for col in cat_columns:
                if col not in encoded_cats.columns:
                    encoded_cats[col] = 0
            # Keep only the columns that were present during training
            encoded_cats = encoded_cats[cat_columns]
            # Join with the main dataframe
            encoded_df = pd.concat([encoded_df.drop('blr_category', axis=1), encoded_cats], axis=1)
        except FileNotFoundError:
            print("Warning: Category encoding mapping not found. Using default encoding.")
            encoded_df = pd.get_dummies(encoded_df, columns=['blr_category'], prefix='cat')
    
    # Drop the original date column as we've extracted features from it
    if 'txn_date' in encoded_df.columns:
        encoded_df['date_ordinal'] = encoded_df['txn_date'].apply(lambda x: x.toordinal())
    
    return encoded_df

def prepare_features_for_modeling(df, target_column, prediction_type='amount'):
    """
    Prepare final feature set for modeling
    """
    if df is None or df.empty:
        print("No data for feature preparation")
        return None, None
    
    # Make a copy of the dataframe
    model_df = df.copy()
    
    # Create time features
    model_df = create_time_features(model_df)
    
    # Create historical features
    model_df = create_historical_features(model_df)
    
    # Save original categorical values and dates before encoding
    category_values = model_df['blr_category'].copy()
    date_values = model_df['txn_date'].copy()
    
    # Encode categorical features (with training mode active)
    model_df = encode_categorical_features(model_df, training_mode=True)
    
    # Select relevant features based on prediction type
    if prediction_type == 'amount':
        # For transaction amount prediction
        feature_columns = [col for col in model_df.columns if col != target_column and col != 'txn_date']
    else:
        # For transaction volume prediction
        feature_columns = [col for col in model_df.columns if col != target_column and col != 'txn_date']
    
    # Create feature matrix and target vector
    X = model_df[feature_columns]
    y = model_df[target_column]
    
    # Add back metadata for later reference (but this won't be used by the model)
    X['original_category'] = category_values.values
    X['original_date'] = date_values.values
    
    # Save clean feature list (without metadata columns)
    clean_features = [col for col in feature_columns if col not in ['original_category', 'original_date']]
    
    # Save the clean feature list
    if prediction_type == 'amount':
        joblib.dump(clean_features, 'models/amount_model_features.joblib')
    else:
        joblib.dump(clean_features, 'models/volume_model_features.joblib')
    
    return X, y