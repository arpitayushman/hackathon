import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import logging

class TransactionPredictor:
    def __init__(self, 
                 models_path='models/transaction_models.pkl', 
                 log_level=logging.INFO):
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Create log directory
        os.makedirs('logs', exist_ok=True)
        
        # Create log file handler
        log_file = f'logs/continuous_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        
        # Load models
        try:
            self.models = joblib.load(models_path)
            self.logger.info(f"Models loaded from {models_path}")
            
            # Extract models and preprocessing components
            self.value_model = self.models['value_model']['model']
            self.volume_model = self.models['volume_model']['model']
            self.value_scaler = self.models['value_model']['scaler']
            self.volume_scaler = self.models['volume_model']['scaler']
            self.label_encoders = self.models['label_encoders']
            
            self.logger.info("Model components extracted successfully")
        except FileNotFoundError:
            self.logger.error("Model file not found. Please run train.py first.")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            sys.exit(1)
    
    def prepare_input_data(self, 
                            date, 
                            category=None, 
                            biller=None, 
                            cou_id=None):
        """
        Prepare input data for prediction
        """
        # Parse input date
        parsed_date = pd.to_datetime(date)
        
        # Validate and handle inputs
        # Categories
        available_categories = self.label_encoders['blr_category'].classes_
        if category is None:
            category = available_categories[0]
        elif category not in available_categories:
            category = available_categories[0]
        
        # COU IDs
        available_cou_ids = self.label_encoders['cou_id'].classes_
        if cou_id is None:
            cou_id = available_cou_ids[0]
        elif cou_id not in available_cou_ids:
            cou_id = available_cou_ids[0]
        
        # Encode categorical variables
        category_encoded = self.label_encoders['blr_category'].transform([category])[0]
        cou_id_encoded = self.label_encoders['cou_id'].transform([cou_id])[0]
        
        # Prepare input features
        input_features = [
            category_encoded,  # blr_category_encoded
            0,  # payment_channel_encoded (default)
            cou_id_encoded,  # cou_id_encoded
            0,  # status_encoded (default)
            parsed_date.month,  # month
            parsed_date.dayofweek,  # day_of_week
            0  # txn_amount (placeholder, will be scaled)
        ]
        
        return np.array(input_features).reshape(1, -1)
    
    def predict_continuous(self, 
                            end_date, 
                            category=None, 
                            biller=None, 
                            cou_id=None):
        """
        Predict transaction values and volumes continuously from today to end date
        """
        self.logger.info(f"Starting continuous prediction until {end_date}")
        
        # Start from today
        start_date = datetime.now().date()
        end_date = pd.to_datetime(end_date).date()
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Prepare prediction results
        predictions = []
        
        for date in date_range:
            try:
                # Prepare input data
                input_data = self.prepare_input_data(
                    date, category, biller, cou_id
                )
                
                # Scale input data
                input_data_scaled_value = self.value_scaler.transform(input_data)
                input_data_scaled_volume = self.volume_scaler.transform(input_data)
                
                # Predict value and volume
                predicted_value = self.value_model.predict(input_data_scaled_value)[0]
                predicted_volume = self.volume_model.predict(input_data_scaled_volume)[0]
                
                # Store prediction
                predictions.append({
                    'date': date,
                    'category': category or 'Default',
                    'biller': biller or 'N/A',
                    'cou_id': cou_id or 'Default',
                    'predicted_value': predicted_value,
                    'predicted_volume': predicted_volume
                })
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {date}: {e}")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Visualize predictions
        self.visualize_continuous_predictions(predictions_df)
        
        return predictions_df
    
    def visualize_continuous_predictions(self, predictions_df):
        """
        Create visualizations for continuous predictions
        """
        # Create output directory
        os.makedirs('predictions', exist_ok=True)
        
        # Plotting predictions
        plt.figure(figsize=(16, 6))
        
        # Transaction Value Plot
        plt.subplot(1, 2, 1)
        plt.plot(predictions_df['date'], predictions_df['predicted_value'], 
                 marker='o', linestyle='-', linewidth=2, markersize=5)
        plt.title('Predicted Transaction Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Predicted Transaction Value')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Transaction Volume Plot
        plt.subplot(1, 2, 2)
        plt.plot(predictions_df['date'], predictions_df['predicted_volume'], 
                 marker='o', linestyle='-', linewidth=2, markersize=5, 
                 color='green')
        plt.title('Predicted Transaction Volumes Over Time')
        plt.xlabel('Date')
        plt.ylabel('Predicted Transaction Volume')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save predictions to CSV
        filename_base = f'continuous_prediction_{predictions_df["date"].min().strftime("%Y%m%d")}_{predictions_df["date"].max().strftime("%Y%m%d")}'
        csv_path = f'predictions/{filename_base}.csv'
        predictions_df.to_csv(csv_path, index=False)
        self.logger.info(f"Prediction CSV saved to {csv_path}")
        
        # Save plot
        plot_path = f'predictions/{filename_base}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {plot_path}")

def main():
    # Configure logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger('main')
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Continuous Transaction Prediction Tool')
    
    # End date is mandatory
    parser.add_argument('end_date', type=str, 
                        help='End date for continuous prediction (YYYY-MM-DD format)')
    
    # Optional arguments
    parser.add_argument('-c', '--category', type=str, default=None,
                        help='Biller category (optional)')
    parser.add_argument('-b', '--biller', type=str, default=None,
                        help='Specific biller (optional)')
    parser.add_argument('-cou', '--cou_id', type=str, default=None,
                        help='COU ID (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = TransactionPredictor()
        
        # Predict continuously
        predictions = predictor.predict_continuous(
            end_date=args.end_date,
            category=args.category,
            biller=args.biller,
            cou_id=args.cou_id
        )
        
        # Print prediction results summary
        logger.info("Continuous Prediction Results Summary:")
        logger.info(f"Prediction Period: {predictions['date'].min()} to {predictions['date'].max()}")
        logger.info(f"Total Days Predicted: {len(predictions)}")
        logger.info(f"Average Predicted Value: {predictions['predicted_value'].mean():.2f}")
        logger.info(f"Average Predicted Volume: {predictions['predicted_volume'].mean():.2f}")
    
    except Exception as e:
        logger.error(f"An error occurred during continuous prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()