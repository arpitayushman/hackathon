import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import argparse
from utils import load_data, preprocess_data, engineer_features, prepare_modeling_data, create_output_dirs

warnings.filterwarnings('ignore')

def train_model(X_train, y_train, param_grid=None):
    """Train a Random Forest Regressor model with improved hyperparameter tuning"""
    # Enhanced parameter grid for better exploration
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    # Create the base model
    rf = RandomForestRegressor(random_state=42)
    
    # Create the grid search with 5-fold CV for better stability
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # Increased from 3 to 5
        n_jobs=-1,
        verbose=1,
        scoring='neg_mean_squared_error'
    )
    
    # Fit the grid search model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def evaluate_model(model, X_test, y_test, feature_cols, output_dir):
    """Evaluate the model performance with enhanced visualizations"""
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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    return mse, rmse, mae, r2, feature_importance

def analyze_by_category(df, y_test, y_pred, test_indices, output_dir):
    """Analyze model performance by biller category"""
    # Create a dataframe with actual and predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'blr_category': df.iloc[test_indices]['blr_category'].values
    })
    
    # Calculate metrics by category
    category_metrics = results_df.groupby('blr_category').apply(
        lambda x: pd.Series({
            'mean_actual': x['actual'].mean(),
            'mean_predicted': x['predicted'].mean(),
            'mean_error': (x['predicted'] - x['actual']).mean(),
            'rmse': np.sqrt(mean_squared_error(x['actual'], x['predicted'])),
            'count': len(x)
        })
    ).reset_index()
    
    # Plot mean transaction amount by category
    plt.figure(figsize=(12, 6))
    category_metrics = category_metrics.sort_values('count', ascending=False)
    
    x = np.arange(len(category_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width/2, category_metrics['mean_actual'], width, label='Actual')
    ax.bar(x + width/2, category_metrics['mean_predicted'], width, label='Predicted')
    
    ax.set_ylabel('Mean Transaction Amount')
    ax.set_title('Mean Transaction Amount by Biller Category')
    ax.set_xticks(x)
    ax.set_xticklabels(category_metrics['blr_category'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_performance.png'))
    
    # Save category metrics
    category_metrics.to_csv(os.path.join(output_dir, 'category_metrics.csv'), index=False)
    
    return category_metrics

def save_model(model, encoders, scaler, feature_cols, model_dir, model_name):
    """Save the model and components needed for prediction"""
    model_components = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    model_path = os.path.join(model_dir, f"{model_name}.mdl")
    joblib.dump(model_components, model_path)
    print(f"Model saved as '{model_path}'")
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train transaction amount predictor model')
    parser.add_argument('--data', type=str, default='data/sample.csv', help='Path to the CSV data file')
    parser.add_argument('--output', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model file')
    parser.add_argument('--model_name', type=str, default='transaction_classifier', help='Name of the model file')
    args = parser.parse_args()
    
    # Create output directories
    create_output_dirs()
    
    # Load data
    df = load_data(args.data)
    if df is None:
        return
    
    # Explore data
    print("\nData Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Preprocess data
    df_processed, encoders = preprocess_data(df)
    
    # Engineer features
    df_featured = engineer_features(df_processed, encoders)
    
    # Prepare data for modeling
    X_scaled, y, scaler, feature_cols = prepare_modeling_data(df_featured)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_scaled, y, np.arange(len(X_scaled)), test_size=0.2, random_state=42
    )

    # Try multiple models and ensemble them
    print("\nTraining multiple models for ensemble...")
    models = []

    # Train Random Forest
    rf_model = train_model(X_train, y_train)
    models.append(rf_model)

    # Train a Gradient Boosting model
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models.append(gb)

# Create an ensemble prediction
    y_pred_ensemble = np.mean([model.predict(X_test) for model in models], axis=0)

    # Evaluate ensemble
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mse_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    print(f"Ensemble RMSE: {rmse_ensemble:.2f}")
    print(f"Ensemble R²: {r2_ensemble:.2f}")
    
    # # Analyze by category
    # print("\nAnalyzing performance by biller category...")
    # category_metrics = analyze_by_category(df, y_test, model.predict(X_test), test_idx, args.output)
    # print("\nCategory performance metrics:")
    # print(category_metrics.head())
    
    # # Save the model
    # model_path = save_model(model, encoders, scaler, feature_cols, args.model_dir, args.model_name)
    
    # print(f"\nTraining completed successfully! Model saved at {model_path}")
    print(f"Performance metrics and visualizations saved in {args.output}/")

if __name__ == "__main__":
    main()