import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import os

class ModelTrainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def train_model(self, preprocessed_data):
        """Train model using Grid Search and Cross-Validation"""
        X_train = preprocessed_data['X_train']
        y_train = preprocessed_data['y_train']
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize Random Forest Regressor
        rf = RandomForestRegressor(random_state=self.config['training']['random_state'])
        
        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid,
            cv=self.config['training']['cv_folds'],
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error'
        )
        
        # Fit Grid Search
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Evaluation metrics
        cv_results = grid_search.cv_results_
        best_params = grid_search.best_params_
        
        # Save model
        os.makedirs(os.path.dirname(self.config['model']['save_path']), exist_ok=True)
        joblib.dump({
            'model': best_model,
            'best_params': best_params,
            'preprocessor': preprocessed_data
        }, self.config['model']['save_path'])
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'cv_results': cv_results
        }
    
    def evaluate_model(self, preprocessed_data, model_results):
        """Evaluate model performance"""
        X_test = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        best_model = model_results['best_model']
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }