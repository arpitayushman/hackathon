import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
import argparse
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

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
    """Preprocess the data for modeling with improved feature encoding"""
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert timestamp to datetime
    df_processed['crtn_ts'] = pd.to_datetime(df_processed['crtn_ts'])
    
    # Extract features from datetime
    df_processed['month'] = df_processed['crtn_ts'].dt.month
    df_processed['day'] = df_processed['crtn_ts'].dt.day
    df_processed['hour'] = df_processed['crtn_ts'].dt.hour
    df_processed['dayofweek'] = df_processed['crtn_ts'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df_processed['time_of_day'] = df_processed['hour'].apply(lambda x: 
                                                            'morning' if 5 <= x < 12 else
                                                            'afternoon' if 12 <= x < 17 else
                                                            'evening' if 17 <= x < 21 else
                                                            'night')
    
    # Create dictionary for encoders
    encoders = {}
    
    # Use more advanced encoding for categorical variables
    return df_processed, encoders

def engineer_features(df, encoders):
    """Create enhanced features for the model"""
    df_featured = df.copy()
    
    # Create more informative features
    
    # Response code indicators
    df_featured['is_success'] = (df_featured['response_code'] == '000').astype(int)
    df_featured['is_failure'] = ((df_featured['response_code'] != '000') & 
                               (df_featured['response_code'] != '')).astype(int)
    
    # Transaction complexity indicators
    df_featured['is_cross_bank'] = (df_featured['on_us'] == 'N').astype(int)
    
    # Payment method categories
    df_featured['is_cash'] = (df_featured['payment_mode'] == 'Cash').astype(int)
    df_featured['is_card'] = ((df_featured['payment_mode'] == 'Credit_Card') | 
                            (df_featured['payment_mode'] == 'Debit_Card')).astype(int)
    df_featured['is_digital'] = ((df_featured['payment_mode'] == 'Net_Banking') | 
                               (df_featured['payment_mode'] == 'Wallet') | 
                               (df_featured['payment_mode'] == 'IMPS') |
                               (df_featured['payment_mode'] == 'AEPS') |
                               (df_featured['payment_mode'] == 'CBDC')).astype(int)
    
    # Channel type features
    df_featured['is_assisted'] = ((df_featured['payment_channel'] == 'Branch') | 
                                (df_featured['payment_channel'] == 'Agent')).astype(int)
    df_featured['is_self_service'] = ((df_featured['payment_channel'] == 'ATM') | 
                                    (df_featured['payment_channel'] == 'Mobile') |
                                    (df_featured['payment_channel'] == 'POS')).astype(int)
    
    # Create aggregations for each biller category to capture typical transaction patterns
    biller_stats = df_featured.groupby('blr_category')['txn_amount'].agg(['mean', 'median', 'std']).reset_index()
    biller_stats.columns = ['blr_category', 'biller_mean_txn', 'biller_median_txn', 'biller_std_txn']
    
    # Merge aggregations back
    df_featured = pd.merge(df_featured, biller_stats, on='blr_category', how='left')
    
    # Calculate z-score of transaction within its category (how unusual is this transaction?)
    df_featured['txn_category_zscore'] = (df_featured['txn_amount'] - df_featured['biller_mean_txn']) / df_featured['biller_std_txn'].replace(0, 1)
    
    # Fill missing values
    df_featured = df_featured.fillna(0)
    
    return df_featured

def prepare_modeling_data(df):
    """Prepare features and target using pipelines for better preprocessing"""
    # Define categorical and numerical columns
    categorical_cols = ['txn_type', 'mti', 'blr_category', 'response_code', 
                      'payment_channel', 'cou_id', 'bou_id', 'bou_status',
                      'payment_mode', 'on_us', 'time_of_day']
    
    # Make sure all categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'unknown'
    
    numerical_cols = ['month', 'day', 'hour', 'dayofweek', 'is_weekend',
                     'is_success', 'is_failure', 'is_cross_bank',
                     'is_cash', 'is_card', 'is_digital',
                     'is_assisted', 'is_self_service',
                     'biller_mean_txn', 'biller_median_txn', 'biller_std_txn',
                     'txn_category_zscore']
    
    # Make sure all numerical columns exist
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Select target
    y = df['txn_amount']
    
    # Select features
    X = df[categorical_cols + numerical_cols]
    
    # We don't need to return a scaler or encoders because we'll use a pipeline
    feature_cols = categorical_cols + numerical_cols
    
    return X, y, feature_cols

def create_model_pipeline(model_type='rf'):
    """Create a full preprocessing and model pipeline"""
    # Define categorical and numerical columns
    categorical_cols = ['txn_type', 'mti', 'blr_category', 'response_code', 
                      'payment_channel', 'cou_id', 'bou_id', 'bou_status',
                      'payment_mode', 'on_us', 'time_of_day']
    
    numerical_cols = ['month', 'day', 'hour', 'dayofweek', 'is_weekend',
                     'is_success', 'is_failure', 'is_cross_bank',
                     'is_cash', 'is_card', 'is_digital',
                     'is_assisted', 'is_self_service',
                     'biller_mean_txn', 'biller_median_txn', 'biller_std_txn',
                     'txn_category_zscore']
    
    # Create preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Select model based on type
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(random_state=42)
    elif model_type == 'lasso':
        model = Lasso(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def train_model(X_train, y_train, model_type='rf'):
    """Train model with enhanced hyperparameter tuning using RandomizedSearchCV"""
    # Create the pipeline
    pipeline = create_model_pipeline(model_type)
    
    # Define parameter grid based on model type
    if model_type == 'rf':
        param_grid = {
            'model__n_estimators': randint(100, 500),
            'model__max_depth': randint(10, 50),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'gb':
        param_grid = {
            'model__n_estimators': randint(100, 500),
            'model__learning_rate': uniform(0.01, 0.3),
            'model__max_depth': randint(3, 10),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__subsample': uniform(0.7, 0.3)
        }
    elif model_type in ['ridge', 'lasso']:
        param_grid = {
            'model__alpha': uniform(0.01, 10)
        }
    else:  # linear regression has no hyperparameters to tune
        param_grid = {}
    
    # If there are parameters to tune, use RandomizedSearchCV
    if param_grid:
        print(f"Tuning hyperparameters for {model_type} model...")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        # Fit the grid search
        search.fit(X_train, y_train)
        
        # Get best model
        best_model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        
        return best_model
    else:
        # Just fit the pipeline
        pipeline.fit(X_train, y_train)
        return pipeline

def evaluate_model(model, X_test, y_test, feature_cols, output_dir):
    """Evaluate model performance with enhanced visualizations"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot actual vs predicted values with improved visualization
    plt.figure(figsize=(10, 6))
    
    # Add scatter plot with alpha for density visualization
    plt.scatter(y_test, y_pred, alpha=0.4, color='#1f77b4', edgecolor='none')
    
    # Get the limits for the plot
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    # Add the ideal line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Prediction')
    
    # Add regression line to show trend
    coeffs = np.polyfit(y_test, y_pred, 1)
    plt.plot(np.array([min_val, max_val]), np.polyval(coeffs, np.array([min_val, max_val])), 
             'b-', linewidth=1.5, label='Regression Line')
    
    plt.xlabel('Actual Transaction Amount')
    plt.ylabel('Predicted Transaction Amount')
    plt.title('Actual vs Predicted Transaction Amounts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    
    # Create residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals, alpha=0.4, color='#2ca02c', edgecolor='none')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Transaction Amount')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'))
    
    # For tree-based models, extract feature importances if available
    feature_importance = None
    
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
        # For pipeline with feature names
        feature_names = list(model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out()) + \
                       list(model.named_steps['preprocessor'].transformers_[0][2])
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))

        return mse, rmse, mae, r2, feature_importance
    
    
