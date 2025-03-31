import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os

class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def load_data(self):
        """Load raw transaction data"""
        data_path = self.config['data']['raw_data_path']
        df = pd.read_csv(data_path)
        
        # Convert date column
        df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])
        
        return df
    
    def preprocess_data(self, df):
        """Perform data preprocessing"""
        # Handle missing values
        df['complaince_cd'].fillna('NONE', inplace=True)
        df['complaince_reason'].fillna('NONE', inplace=True)
        
        # Encode categorical variables
        categorical_cols = self.config['preprocessing']['categorical_columns']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Feature engineering
        df['month'] = df['crtn_ts'].dt.month
        df['day_of_week'] = df['crtn_ts'].dt.dayofweek
        
        # Extract numerical features
        feature_cols = [
            col + '_encoded' for col in categorical_cols
        ] + ['month', 'day_of_week', 'txn_amount']
        
        # Prepare features and target
        X = df[feature_cols]
        y_value = df['txn_amount']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_value, 
            test_size=self.config['training']['test_size'], 
            random_state=self.config['training']['random_state']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save processed data
        os.makedirs(os.path.dirname(self.config['data']['processed_train_path']), exist_ok=True)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoders': label_encoders
        }