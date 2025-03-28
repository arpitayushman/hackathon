import pandas as pd
import numpy as np
import joblib
import yaml
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TransactionModelTrainer:
    def __init__(self, config_path='config.yaml', log_level=logging.INFO):
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Create log directory
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler
        log_file = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.logger.info("TransactionModelTrainer initialized")
    
    def load_and_preprocess_data(self):
        """Load and preprocess transaction data"""
        self.logger.info("Starting data loading and preprocessing")
        
        try:
            # Load data
            df = pd.read_csv(self.config['data']['raw_data_path'])
            self.logger.info(f"Data loaded from {self.config['data']['raw_data_path']}")
            self.logger.info(f"Dataset shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
        
        try:
            # Convert date column
            df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])
            self.logger.info("Date column converted to datetime")
        except Exception as e:
            self.logger.error(f"Error converting date column: {e}")
            raise
        
        # Handle missing values
        df['complaince_cd'].fillna('NONE', inplace=True)
        df['complaince_reason'].fillna('NONE', inplace=True)
        self.logger.info("Missing values handled")
        
        # Encode categorical variables
        categorical_cols = ['blr_category', 'payment_channel', 'cou_id', 'status']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            self.logger.info(f"Encoded categorical column: {col}")
            self.logger.info(f"Unique values in {col}: {len(le.classes_)}")
        
        # Feature engineering
        df['month'] = df['crtn_ts'].dt.month
        df['day_of_week'] = df['crtn_ts'].dt.dayofweek
        self.logger.info("Feature engineering completed")
        
        # Prepare features
        feature_cols = [
            col + '_encoded' for col in categorical_cols
        ] + ['month', 'day_of_week', 'txn_amount']
        
        # Prepare features and target
        X = df[feature_cols]
        y_value = df['txn_amount']
        y_volume = df.groupby('blr_category')['ref_id'].transform('count')
        
        self.logger.info("Data preprocessing completed")
        
        return {
            'X': X,
            'y_value': y_value,
            'y_volume': y_volume,
            'label_encoders': label_encoders
        }
    
    def train_model(self, preprocessed_data):
        """Train models for transaction value and volume"""
        self.logger.info("Starting model training")
        
        X = preprocessed_data['X']
        y_value = preprocessed_data['y_value']
        y_volume = preprocessed_data['y_volume']
        label_encoders = preprocessed_data['label_encoders']
        
        # Split data
        X_train_value, X_test_value, y_train_value, y_test_value = train_test_split(
            X, y_value, test_size=0.2, random_state=42
        )
        X_train_volume, X_test_volume, y_train_volume, y_test_volume = train_test_split(
            X, y_volume, test_size=0.2, random_state=42
        )
        
        self.logger.info(f"Data split - Value prediction train size: {X_train_value.shape}")
        self.logger.info(f"Data split - Volume prediction train size: {X_train_volume.shape}")
        
        # Scale features
        scaler_value = StandardScaler()
        X_train_value_scaled = scaler_value.fit_transform(X_train_value)
        X_test_value_scaled = scaler_value.transform(X_test_value)
        
        scaler_volume = StandardScaler()
        X_train_volume_scaled = scaler_volume.fit_transform(X_train_volume)
        X_test_volume_scaled = scaler_volume.transform(X_test_volume)
        
        self.logger.info("Features scaled")
        
        # Value Prediction Model
        rf_value = RandomForestRegressor(random_state=42)
        param_grid_value = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        self.logger.info("Grid Search - Value Prediction")
        self.logger.info(f"Parameter Grid: {param_grid_value}")
        self.logger.info(f"CV Folds: 5")
        
        grid_search_value = GridSearchCV(
            estimator=rf_value, 
            param_grid=param_grid_value,
            cv=5,
            scoring='neg_mean_squared_error',
            refit=True,
            verbose=1  # Show grid search progress
        )
        grid_search_value.fit(X_train_value_scaled, y_train_value)
        
        # Volume Prediction Model
        rf_volume = RandomForestRegressor(random_state=42)
        param_grid_volume = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        self.logger.info("Grid Search - Volume Prediction")
        self.logger.info(f"Parameter Grid: {param_grid_volume}")
        self.logger.info(f"CV Folds: 5")
        
        grid_search_volume = GridSearchCV(
            estimator=rf_volume, 
            param_grid=param_grid_volume,
            cv=5,
            scoring='neg_mean_squared_error',
            refit=True,
            verbose=1  # Show grid search progress
        )
        grid_search_volume.fit(X_train_volume_scaled, y_train_volume)
        
        # Log best parameters
        self.logger.info("Best Parameters - Value Prediction:")
        self.logger.info(grid_search_value.best_params_)
        
        self.logger.info("Best Parameters - Volume Prediction:")
        self.logger.info(grid_search_volume.best_params_)
        
        # Prepare model artifacts
        model_artifacts = {
            'value_model': {
                'model': grid_search_value.best_estimator_,
                'scaler': scaler_value,
                'best_params': grid_search_value.best_params_
            },
            'volume_model': {
                'model': grid_search_volume.best_estimator_,
                'scaler': scaler_volume,
                'best_params': grid_search_volume.best_params_
            },
            'label_encoders': label_encoders
        }
        
        # Evaluate models
        value_metrics = self.evaluate_model(
            grid_search_value.best_estimator_, 
            scaler_value, 
            X_test_value, 
            y_test_value
        )
        volume_metrics = self.evaluate_model(
            grid_search_volume.best_estimator_, 
            scaler_volume, 
            X_test_volume, 
            y_test_volume
        )
        
        # Log evaluation metrics
        self.logger.info("Value Prediction Metrics:")
        for metric, value in value_metrics.items():
            self.logger.info(f"{metric}: {value}")
        
        self.logger.info("Volume Prediction Metrics:")
        for metric, value in volume_metrics.items():
            self.logger.info(f"{metric}: {value}")
        
        # Save models and metrics
        os.makedirs('models', exist_ok=True)
        try:
            joblib.dump(model_artifacts, 'models/transaction_models.pkl')
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
        
        return {
            'value_metrics': value_metrics,
            'volume_metrics': volume_metrics
        }
    
    def evaluate_model(self, model, scaler, X_test, y_test):
        """Evaluate model performance"""
        self.logger.info("Evaluating model performance")
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

def main():
    # Configure logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('main')
    
    try:
        trainer = TransactionModelTrainer()
        logger.info("Starting data preprocessing")
        preprocessed_data = trainer.load_and_preprocess_data()
        
        logger.info("Starting model training")
        metrics = trainer.train_model(preprocessed_data)
        
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()