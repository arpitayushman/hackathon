import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime

class TransactionPredictor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load saved model and preprocessor
        model_data = joblib.load(self.config['model']['save_path'])
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.label_encoders = self.preprocessor['label_encoders']
    
    def predict(self, 
                date, 
                category=None, 
                cou_id=None):
        """
        Predict transaction value and volume
        
        :param date: Date for prediction (str in yyyy-MM-dd or yyyy-MM format)
        :param category: Optional biller category
        :param cou_id: Optional COU ID
        :return: Dictionary with predictions
        """
        # Parse input date
        try:
            parsed_date = pd.to_datetime(date)
        except ValueError:
            raise ValueError("Invalid date format. Use yyyy-MM-dd or yyyy-MM")
        
        # Prepare prediction input
        input_data = {
            'month': parsed_date.month,
            'day_of_week': parsed_date.dayofweek
        }
        
        # Handle category
        if category is None:
            category = self.config['prediction']['default_category']
        category_encoded = self.label_encoders['blr_category'].transform([category])[0]
        input_data['blr_category_encoded'] = category_encoded
        
        # Handle COU ID
        if cou_id is None:
            cou_id = self.config['prediction']['default_cou_id']
        cou_id_encoded = self.label_encoders['cou_id'].transform([cou_id])[0]
        input_data['cou_id_encoded'] = cou_id_encoded
        
        # Add placeholder for transaction amount (will be scaled)
        input_data['txn_amount'] = np.mean(self.preprocessor['y_train'])
        
        # Prepare input for prediction
        input_array = np.array([
            input_data['blr_category_encoded'],
            input_data['cou_id_encoded'],
            input_data['month'],
            input_data['day_of_week'],
            input_data['txn_amount']
        ]).reshape(1, -1)
        
        # Scale input
        input_scaled = self.preprocessor['scaler'].transform(input_array)
        
        # Predict transaction value
        predicted_value = self.model.predict(input_scaled)[0]
        
        # Predict transaction volume (based on historical data)
        # This is a simplified approach and might need more sophisticated modeling
        volume_estimate = len(self.preprocessor['y_train']) // len(set(self.preprocessor['y_train']))
        
        return {
            'transaction_value': predicted_value,
            'transaction_volume': volume_estimate
        }