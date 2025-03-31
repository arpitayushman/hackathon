import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets, respecting time order
    """
    # Get unique dates and sort them
    dates = X['original_date'].unique()
    dates = sorted(dates)  # Replace dates.sort() with sorted(dates)
    
    # Calculate split point
    split_idx = int(len(dates) * (1 - test_size))
    split_date = dates[split_idx]
    
    # Split data based on date
    train_idx = X['original_date'] < split_date
    test_idx = X['original_date'] >= split_date
    
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx].copy()
    y_test = y[test_idx].copy()
    
    # Store the full versions with metadata for reference
    X_train_full = X_train.copy()
    X_test_full = X_test.copy()
    
    # Remove metadata columns for actual training
    meta_cols = ['original_category', 'original_date']
    X_train_clean = X_train.drop(columns=meta_cols)
    X_test_clean = X_test.drop(columns=meta_cols)
    
    print(f"Training data: {X_train_clean.shape[0]} samples")
    print(f"Testing data: {X_test_clean.shape[0]} samples")
    
    return X_train_clean, X_test_clean, y_train, y_test, X_train_full, X_test_full

def train_amount_model(X, y):
    """
    Train a model to predict transaction amount
    """
    # Save feature names for later use in prediction
    joblib.dump(X.columns.tolist(), 'models/amount_model_features.joblib')

    # Split data
    X_train, X_test, y_train, y_test, X_train_full, X_test_full = split_data(X, y)
    
    # Initialize and train models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'linear_regression': LinearRegression()
    }
    
    results = {}
    best_model = None
    best_score = float('inf')  # Lower is better for MAE
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        # rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Track best model
        if mae < best_score:
            best_score = mae
            best_model = model
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/amount_model.joblib')
    
    # Save feature importance for best model if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('models/amount_feature_importance.csv', index=False)
    
    return results, best_model

def train_volume_model(X, y):
    """
    Train a model to predict transaction volume
    """
    # Save feature names for later use in prediction
    joblib.dump(X.columns.tolist(), 'models/volume_model_features.joblib')

    # Split data
    X_train, X_test, y_train, y_test, X_train_full, X_test_full = split_data(X, y)
    
    # Initialize and train models
    models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(random_state=42),
        'linear_regression': LinearRegression()
    }
    
    results = {}
    best_model = None
    best_score = float('inf')  # Lower is better for MAE
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        # rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model
        }
        
        print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Track best model
        if mae < best_score:
            best_score = mae
            best_model = model
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/volume_model.joblib')
    
    # Save feature importance for best model if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('models/volume_feature_importance.csv', index=False)
    
    return results, best_model

def optimize_hyperparameters(X, y, model_type='amount'):
    """
    Perform hyperparameter optimization for the best model
    """
    # Split data
    X_train, X_test, y_train, y_test, X_train_full, X_test_full = split_data(X, y)
    
    # Define model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform grid search
    print("Performing hyperparameter optimization...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                             cv=tscv, scoring='neg_mean_absolute_error',
                             n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters: {best_params}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Optimized model - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Save optimized model
    model_filename = f'models/optimized_{model_type}_model.joblib'
    joblib.dump(best_model, model_filename)
    
    # Save optimization results
    optimization_results = {
        'best_params': best_params,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_results': grid_search.cv_results_
    }
    
    joblib.dump(optimization_results, f'models/optimization_results_{model_type}.joblib')
    
    # Save feature importance for optimized model
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv(f'models/optimized_{model_type}_feature_importance.csv', index=False)
    
    return best_model, optimization_results

def evaluate_model_performance(model, X, y, model_type='amount'):
    """
    Evaluate model performance on different segments of data
    """
    # Split data
    X_train, X_test, y_train, y_test, X_train_full, X_test_full = split_data(X, y)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Overall performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nOverall {model_type} model performance:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Performance by biller category
    categories = X_test_full['original_category'].unique()
    category_performance = {}
    
    for category in categories:
        cat_mask = X_test_full['original_category'] == category
        if sum(cat_mask) > 0:  # Only evaluate if we have data for this category
            cat_X = X_test[cat_mask]
            cat_y = y_test[cat_mask]
            cat_y_pred = model.predict(cat_X)
            
            cat_mae = mean_absolute_error(cat_y, cat_y_pred)
            cat_rmse = mean_squared_error(cat_y, cat_y_pred, squared=False)
            cat_r2 = r2_score(cat_y, cat_y_pred) if len(cat_y) > 1 else 0
            
            category_performance[category] = {
                'mae': cat_mae,
                'rmse': cat_rmse,
                'r2': cat_r2,
                'sample_count': sum(cat_mask)
            }
    
    # Save performance by category
    cat_perf_df = pd.DataFrame.from_dict(category_performance, orient='index')
    cat_perf_df = cat_perf_df.sort_values('sample_count', ascending=False)
    cat_perf_df.to_csv(f'models/{model_type}_category_performance.csv')
    
    print(f"\nPerformance by category saved to models/{model_type}_category_performance.csv")
    
    # Return evaluation results
    return {
        'overall': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'by_category': category_performance
    }