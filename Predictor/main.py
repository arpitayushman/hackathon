import yaml
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.prediction import TransactionPredictor

def main():
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Data Preprocessing
    preprocessor = DataPreprocessor()
    raw_data = preprocessor.load_data()
    preprocessed_data = preprocessor.preprocess_data(raw_data)
    
    # Model Training
    trainer = ModelTrainer()
    model_results = trainer.train_model(preprocessed_data)
    
    # Model Evaluation
    evaluation_metrics = trainer.evaluate_model(preprocessed_data, model_results)
    print("Model Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")
    
    # Example Prediction
    predictor = TransactionPredictor()
    
    # Predict for a specific date, category, and COU ID
    prediction = predictor.predict(
        date='2025-03-15', 
        category='Water', 
        cou_id='GP01'
    )
    
    print("\nPrediction Results:")
    print(f"Estimated Transaction Value: {prediction['transaction_value']}")
    print(f"Estimated Transaction Volume: {prediction['transaction_volume']}")

if __name__ == "__main__":
    main()