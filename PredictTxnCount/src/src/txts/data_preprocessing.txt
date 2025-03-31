import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """
    Load transaction data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean and preprocess the transaction data
    """
    if df is None or df.empty:
        print("No data to clean")
        return None
    
    # Make a copy to avoid modifying original data
    cleaned_df = df.copy()
    
    # Convert transaction date to datetime
    cleaned_df['crtn_ts'] = pd.to_datetime(cleaned_df['crtn_ts'])
    
    # Extract date components
    cleaned_df['txn_date'] = cleaned_df['crtn_ts'].dt.date
    cleaned_df['year'] = cleaned_df['crtn_ts'].dt.year
    cleaned_df['month'] = cleaned_df['crtn_ts'].dt.month
    cleaned_df['day'] = cleaned_df['crtn_ts'].dt.day
    cleaned_df['day_of_week'] = cleaned_df['crtn_ts'].dt.dayofweek
    cleaned_df['is_weekend'] = cleaned_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Handle missing values
    for col in cleaned_df.columns:
        # For numeric columns, fill missing values with median
        if cleaned_df[col].dtype in ['int64', 'float64'] and cleaned_df[col].isna().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        # For categorical columns, fill missing values with mode
        elif cleaned_df[col].dtype == 'object' and cleaned_df[col].isna().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
    
    # Convert transaction amount to float if needed
    if 'txn_amount' in cleaned_df.columns and cleaned_df['txn_amount'].dtype != 'float64':
        cleaned_df['txn_amount'] = pd.to_numeric(cleaned_df['txn_amount'], errors='coerce')
        cleaned_df['txn_amount'] = cleaned_df['txn_amount'].fillna(cleaned_df['txn_amount'].median())
    
    # Filter out any data with future dates beyond our knowledge cutoff
    today = datetime.now().date()
    cleaned_df = cleaned_df[cleaned_df['txn_date'] <= today]
    
    print(f"Data cleaning completed. Cleaned data has {cleaned_df.shape[0]} rows.")
    return cleaned_df

def create_analysis_data(df):
    """
    Create aggregated datasets for analysis and modeling
    """
    if df is None or df.empty:
        print("No data for analysis")
        return None, None
    
    # Create a date range for all dates
    min_date = df['txn_date'].min()
    max_date = df['txn_date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Get unique biller categories
    biller_categories = df['blr_category'].unique()
    
    # Create transaction amount dataset aggregated by date and biller category
    amount_df = df.groupby(['txn_date', 'blr_category'])['txn_amount'].agg(['mean', 'sum', 'count']).reset_index()
    amount_df.columns = ['txn_date', 'blr_category', 'avg_amount', 'total_amount', 'txn_count']
    
    # Add year, month, day features
    amount_df['txn_date'] = pd.to_datetime(amount_df['txn_date'])
    amount_df['year'] = amount_df['txn_date'].dt.year
    amount_df['month'] = amount_df['txn_date'].dt.month
    amount_df['day'] = amount_df['txn_date'].dt.day
    amount_df['day_of_week'] = amount_df['txn_date'].dt.dayofweek
    amount_df['is_weekend'] = amount_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create complete date x category dataset
    date_cat_combinations = []
    for date in all_dates:
        for cat in biller_categories:
            date_cat_combinations.append((date.date(), cat))
    
    complete_df = pd.DataFrame(date_cat_combinations, columns=['txn_date', 'blr_category'])
    complete_df['txn_date'] = pd.to_datetime(complete_df['txn_date'])
    
    # Merge with actual data
    complete_df = pd.merge(complete_df, amount_df, on=['txn_date', 'blr_category'], how='left')
    
    # Fill missing values
    complete_df['avg_amount'] = complete_df.groupby('blr_category')['avg_amount'].transform(
        lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
    )
    
    complete_df['total_amount'] = complete_df.groupby('blr_category')['total_amount'].transform(
        lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
    )
    
    complete_df['txn_count'] = complete_df.groupby('blr_category')['txn_count'].transform(
        lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0)
    )
    
    # Add time components if not present
    if 'year' not in complete_df.columns:
        complete_df['year'] = complete_df['txn_date'].dt.year
        complete_df['month'] = complete_df['txn_date'].dt.month
        complete_df['day'] = complete_df['txn_date'].dt.day
        complete_df['day_of_week'] = complete_df['txn_date'].dt.dayofweek
        complete_df['is_weekend'] = complete_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create separate dataframes for amount prediction and volume prediction
    amount_prediction_df = complete_df.copy()
    volume_prediction_df = complete_df.copy()
    
    print(f"Analysis data created with {complete_df.shape[0]} rows covering {len(biller_categories)} categories")
    return amount_prediction_df, volume_prediction_df