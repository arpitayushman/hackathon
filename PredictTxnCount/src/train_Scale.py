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
from datetime import datetime
import calendar

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_output_dirs(output_dir='output', model_dir='models'):
    """Create output directories if they don't exist"""
    for directory in [output_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def preprocess_data(df):
    """Preprocess data for transaction count prediction"""
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Use the specific date column 'crtn_ts' instead of searching for date columns
    date_col = 'crtn_ts'
    
    if date_col in df.columns:
        df_processed[date_col] = pd.to_datetime(df_processed[date_col])
        
        # Extract date features
        df_processed['year'] = df_processed[date_col].dt.year
        df_processed['month'] = df_processed[date_col].dt.month
        df_processed['day'] = df_processed[date_col].dt.day
        df_processed['day_of_week'] = df_processed[date_col].dt.dayofweek
        df_processed['is_weekend'] = df_processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_processed['is_month_start'] = df_processed[date_col].dt.is_month_start.astype(int)
        df_processed['is_month_end'] = df_processed[date_col].dt.is_month_end.astype(int)
    else:
        raise ValueError(f"Required column '{date_col}' not found in dataframe")
    
    # Handle missing values in important columns
    if 'blr_category' in df_processed.columns:
        df_processed['blr_category'].fillna('Unknown', inplace=True)
    
    # Store encoders for categorical variables
    encoders = {}
    
    return df_processed, encoders

def aggregate_transaction_counts(df):
    """Aggregate data to get transaction counts by biller category and time period"""
    # Ensure we have date and biller category columns
    if 'blr_category' not in df.columns:
        raise ValueError("DataFrame must contain 'blr_category' column")
    
    date_col = 'crtn_ts'
    if date_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{date_col}' column")
    
    # Create a time-based identifier for grouping
    # This will aggregate by year and month
    df['time_id'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    # Count unique ref_ids per biller category per time period
    if 'ref_id' in df.columns:
        txn_counts = df.groupby(['time_id', 'year', 'month', 'blr_category'])['ref_id'].nunique().reset_index()
        txn_counts.rename(columns={'ref_id': 'txn_count'}, inplace=True)
    else:
        # If no ref_id column, count rows
        txn_counts = df.groupby(['time_id', 'year', 'month', 'blr_category']).size().reset_index(name='txn_count')
    
    # Sort by time and category
    txn_counts = txn_counts.sort_values(['time_id', 'blr_category'])
    
    return txn_counts

def prepare_modeling_data(df_featured):
    """Prepare the final dataset for modeling"""
    # Drop non-feature columns
    feature_cols = df_featured.columns.tolist()
    for col in ['time_id', 'blr_category', 'month_name', 'txn_count']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    # Define X and y
    X = df_featured[feature_cols].copy()
    y = df_featured['txn_count'].copy()
    
    # Scale numeric features if needed
    from sklearn.preprocessing import StandardScaler
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler, feature_cols

def train_model(X_train, y_train, param_grid=None):
    """Train a Random Forest Regressor model with hyperparameter tuning"""
    # Default parameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    
    # Create the base model
    rf = RandomForestRegressor(random_state=42)
    
    # Create the grid search model
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
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
    """Evaluate the model performance"""
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
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Transaction Count')
    plt.ylabel('Predicted Transaction Count')
    plt.title('Actual vs Predicted Transaction Counts')
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
    return mse, rmse, mae, r2, feature_importance

def analyze_by_category(df_featured, y_test, y_pred, test_indices, output_dir):
    """Analyze model performance by biller category"""
    # Create a dataframe with actual and predicted values
    test_df = df_featured.iloc[test_indices].copy()
    results_df = pd.DataFrame({
        'time_id': test_df['time_id'],
        'blr_category': test_df['blr_category'],
        'month': test_df['month'],
        'month_name': test_df['month_name'],
        'actual': y_test,
        'predicted': y_pred
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
    
    # Plot mean transaction count by category
    plt.figure(figsize=(12, 6))
    category_metrics = category_metrics.sort_values('count', ascending=False)
    
    x = np.arange(len(category_metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width/2, category_metrics['mean_actual'], width, label='Actual')
    ax.bar(x + width/2, category_metrics['mean_predicted'], width, label='Predicted')
    
    ax.set_ylabel('Mean Transaction Count')
    ax.set_title('Mean Transaction Count by Biller Category')
    ax.set_xticks(x)
    ax.set_xticklabels(category_metrics['blr_category'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_performance.png'))
    
    # Analyze monthly patterns
    monthly_metrics = results_df.groupby(['month', 'month_name']).apply(
        lambda x: pd.Series({
            'mean_actual': x['actual'].mean(),
            'mean_predicted': x['predicted'].mean(),
            'mean_error': (x['predicted'] - x['actual']).mean(),
            'count': len(x)
        })
    ).reset_index()
    
    # Sort by month number
    monthly_metrics = monthly_metrics.sort_values('month')
    
    # Plot monthly trends
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_metrics['month_name'], monthly_metrics['mean_actual'], 'o-', label='Actual')
    plt.plot(monthly_metrics['month_name'], monthly_metrics['mean_predicted'], 's--', label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Average Transaction Count')
    plt.title('Monthly Transaction Count Patterns')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_patterns.png'))
    
    # Save metrics to CSV
    category_metrics.to_csv(os.path.join(output_dir, 'category_metrics.csv'), index=False)
    monthly_metrics.to_csv(os.path.join(output_dir, 'monthly_metrics.csv'), index=False)
    
    return category_metrics, monthly_metrics

def save_model(model, scaler, feature_cols, model_dir, model_name):
    """Save the model and components needed for prediction"""
    model_components = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
    
    model_path = os.path.join(model_dir, f"{model_name}.mdl")
    joblib.dump(model_components, model_path)
    print(f"Model saved as '{model_path}'")
    return model_path

def generate_future_predictions(model, df_featured, scaler, feature_cols, months_to_predict=12, output_dir='output'):
    """Generate predictions for future months"""
    # Get the latest time id
    latest_time = df_featured['time_id'].max()
    latest_year = df_featured[df_featured['time_id'] == latest_time]['year'].iloc[0]
    latest_month = df_featured[df_featured['time_id'] == latest_time]['month'].iloc[0]
    
    # Create a dataframe for future months
    future_months = []
    year = latest_year
    month = latest_month
    
    # Only use categories that appear in the training data
    # IMPORTANT: Filter categories to only those that have corresponding features in feature_cols
    available_category_cols = [col for col in feature_cols if col.startswith('category_')]
    valid_categories = [col.replace('category_', '') for col in available_category_cols]
    
    print(f"Using {len(valid_categories)} categories for prediction that were found in training data")
    
    # Store the historical data for each category by month
    historical_monthly_data = {}
    for category in valid_categories:
        category_data = df_featured[df_featured['blr_category'] == category].copy()
        if not category_data.empty:
            historical_monthly_data[category] = {}
            for m in range(1, 13):
                month_data = category_data[category_data['month'] == m]
                if not month_data.empty:
                    historical_monthly_data[category][m] = month_data['txn_count'].mean()
    
    # Generate data for future months
    for i in range(1, months_to_predict + 1):
        # Move to next month
        month += 1
        if month > 12:
            month = 1
            year += 1
        
        time_id = f"{year}-{str(month).zfill(2)}"
        
        # For each category, create a row
        for category in valid_categories:
            # Initialize with default values
            future_row = {
                'time_id': time_id,
                'year': year,
                'month': month,
                'blr_category': category,
                'month_name': calendar.month_name[month]
            }
            
            # Only create features that exist in feature_cols
            for col in feature_cols:
                future_row[col] = 0  # Initialize all features to 0
            
            # Set month feature
            month_col = f'month_{month}'
            if month_col in feature_cols:
                future_row[month_col] = 1
            
            # Set category feature
            category_col = f'category_{category}'
            if category_col in feature_cols:
                future_row[category_col] = 1
            
            # Find the previous month's transaction count for this category
            prev_month_idx = month - 1 if month > 1 else 12
            prev_year = year if month > 1 else year - 1
            prev_time_id = f"{prev_year}-{str(prev_month_idx).zfill(2)}"
            
            # Find previous month's count from either predictions or historical data
            prev_month_data = [fm for fm in future_months if fm['time_id'] == prev_time_id and fm['blr_category'] == category]
            
            if prev_month_data and 'predicted_txn_count' in prev_month_data[0]:
                # Use the prediction from the previous month
                if 'prev_month_count' in feature_cols:
                    future_row['prev_month_count'] = prev_month_data[0]['predicted_txn_count']
            else:
                # Use historical data from the latest year
                latest_prev_month = df_featured[(df_featured['blr_category'] == category) & 
                                                (df_featured['month'] == prev_month_idx)]
                
                if 'prev_month_count' in feature_cols:
                    if not latest_prev_month.empty:
                        future_row['prev_month_count'] = latest_prev_month.iloc[-1]['txn_count']
                    else:
                        # If no data for previous month, use category's monthly average
                        if category in historical_monthly_data and prev_month_idx in historical_monthly_data[category]:
                            future_row['prev_month_count'] = historical_monthly_data[category][prev_month_idx]
                        else:
                            # Default to category average
                            cat_data = df_featured[df_featured['blr_category'] == category]
                            future_row['prev_month_count'] = cat_data['txn_count'].mean() if not cat_data.empty else 0
            
            # Find same month last year for this category
            prev_year_same_month_data = df_featured[(df_featured['blr_category'] == category) & 
                                                     (df_featured['month'] == month) & 
                                                     (df_featured['year'] == year - 1)]
            
            if 'prev_year_same_month_count' in feature_cols:
                if not prev_year_same_month_data.empty:
                    future_row['prev_year_same_month_count'] = prev_year_same_month_data.iloc[-1]['txn_count']
                else:
                    # If no data for same month last year, use monthly average for this category
                    if category in historical_monthly_data and month in historical_monthly_data[category]:
                        future_row['prev_year_same_month_count'] = historical_monthly_data[category][month]
                    else:
                        # Default to category average
                        cat_data = df_featured[df_featured['blr_category'] == category]
                        future_row['prev_year_same_month_count'] = cat_data['txn_count'].mean() if not cat_data.empty else 0
            
            future_months.append(future_row)
    
    # Convert to dataframe
    future_df = pd.DataFrame(future_months)
    
    # Ensure we only use the exact feature columns that were used during training
    # This is the critical fix for the feature name mismatch error
    X_future = future_df[feature_cols].copy()
    
    # Scale features
    numeric_cols = X_future.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:  # Only transform if there are numeric columns
        X_future[numeric_cols] = scaler.transform(X_future[numeric_cols])
    
    # Make predictions
    future_predictions = model.predict(X_future)
    
    # Add predictions to the dataframe
    future_df['predicted_txn_count'] = np.maximum(0, future_predictions.round())  # Ensure non-negative
    
    # Plot predictions by category
    plt.figure(figsize=(14, 8))
    
    # Select top categories by predicted transaction count
    top_categories = future_df.groupby('blr_category')['predicted_txn_count'].sum().nlargest(5).index.tolist()
    
    for category in top_categories:
        category_data = future_df[future_df['blr_category'] == category].sort_values('time_id')
        plt.plot(category_data['time_id'], category_data['predicted_txn_count'], marker='o', label=category)
    
    plt.title('Predicted Transaction Counts for Top 5 Biller Categories')
    plt.xlabel('Time Period')
    plt.ylabel('Predicted Transaction Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'future_predictions_by_category.png'))
    
    # Plot seasonal patterns (month-by-month average across categories)
    monthly_avg = future_df.groupby(['month', 'month_name'])['predicted_txn_count'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('month')
    
    plt.figure(figsize=(12, 6))
    plt.bar(monthly_avg['month_name'], monthly_avg['predicted_txn_count'])
    plt.title('Predicted Average Monthly Transaction Counts')
    plt.xlabel('Month')
    plt.ylabel('Average Predicted Transaction Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_prediction_patterns.png'))
    
    # Save predictions to CSV
    future_df.to_csv(os.path.join(output_dir, 'future_predictions.csv'), index=False)
    
    print(f"Generated predictions for {len(future_df)} category-month combinations")
    return future_df

def prepare_features_for_prediction(txn_counts):
    """Prepare features for the transaction count prediction model"""
    df_featured = txn_counts.copy()
    
    # Create categorical month names for better visualization
    df_featured['month_name'] = df_featured['month'].apply(lambda x: calendar.month_name[x])
    
    # One-hot encode months to capture seasonality
    month_dummies = pd.get_dummies(df_featured['month'], prefix='month')
    df_featured = pd.concat([df_featured, month_dummies], axis=1)
    
    # One-hot encode biller categories
    # Instead of using pd.get_dummies which can create different column names based on the data,
    # we'll manually create the dummy variables to ensure consistency
    categories = df_featured['blr_category'].unique()
    
    # Create a DataFrame with one column per category
    category_dummies = pd.DataFrame()
    for category in categories:
        col_name = f'category_{category}'
        category_dummies[col_name] = (df_featured['blr_category'] == category).astype(int)
    
    df_featured = pd.concat([df_featured, category_dummies], axis=1)
    
    # Create lagged features (previous month's counts)
    for category in categories:
        category_data = df_featured[df_featured['blr_category'] == category].copy()
        category_data = category_data.sort_values('time_id')
        category_data['prev_month_count'] = category_data['txn_count'].shift(1)
        category_data['prev_year_same_month_count'] = category_data['txn_count'].shift(12)
        
        # Update the values in the main dataframe
        df_featured.loc[df_featured['blr_category'] == category, 'prev_month_count'] = category_data['prev_month_count']
        df_featured.loc[df_featured['blr_category'] == category, 'prev_year_same_month_count'] = category_data['prev_year_same_month_count']
    
    # Fill missing lag values with mean or median
    for col in ['prev_month_count', 'prev_year_same_month_count']:
        if col in df_featured.columns:
            # Fill with category-specific means
            for category in categories:
                category_mean = df_featured[df_featured['blr_category'] == category][col].mean()
                mask = (df_featured['blr_category'] == category) & (df_featured[col].isna())
                df_featured.loc[mask, col] = category_mean
            
            # Fill any remaining NaNs with overall mean
            df_featured[col].fillna(df_featured[col].mean(), inplace=True)
    
    return df_featured

def main():
    """Main function to run the transaction prediction pipeline"""
    parser = argparse.ArgumentParser(description='Predict transaction counts by biller category')
    parser.add_argument('--data', type=str, required=True, help='Path to the transaction data CSV file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory for model files')
    parser.add_argument('--predict_months', type=int, default=12, help='Number of months to predict')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Create output directories
    create_output_dirs(args.output_dir, args.model_dir)
    
    # Load data
    df = load_data(args.data)
    if df is None:
        return
    
    # Preprocess data
    df_processed, encoders = preprocess_data(df)
    
    # Aggregate transaction counts
    txn_counts = aggregate_transaction_counts(df_processed)
    
    # Prepare features
    df_featured = prepare_features_for_prediction(txn_counts)
    
    # Prepare modeling data
    X, y, scaler, feature_cols = prepare_modeling_data(df_featured)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    test_indices = y_test.index
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    mse, rmse, mae, r2, feature_importance = evaluate_model(
        model, X_test, y_test, feature_cols, args.output_dir
    )
    
    # Analyze by category
    print("Analyzing performance by category...")
    category_metrics, monthly_metrics = analyze_by_category(
        df_featured, y_test, model.predict(X_test), test_indices, args.output_dir
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_model(
        model, scaler, feature_cols, args.model_dir, f"txn_count_model_{timestamp}"
    )
    
    # Generate future predictions
    print(f"Generating predictions for the next {args.predict_months} months...")
    future_predictions = generate_future_predictions(
        model, df_featured, scaler, feature_cols, args.predict_months, args.output_dir
    )
    
    print("\nPipeline completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Future predictions saved to: {os.path.join(args.output_dir, 'future_predictions.csv')}")
    print(f"Visualizations saved to: {args.output_dir}")
    
    # Print summary of future predictions
    print("\nSummary of predicted transaction counts by biller category:")
    category_summary = future_predictions.groupby('blr_category')['predicted_txn_count'].agg(['mean', 'sum']).reset_index()
    category_summary = category_summary.sort_values('sum', ascending=False).head(10)
    print(category_summary)
    
    # Print monthly trends
    print("\nPredicted monthly transaction pattern:")
    monthly_avg = future_predictions.groupby(['month', 'month_name'])['predicted_txn_count'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('month')
    print(monthly_avg[['month_name', 'predicted_txn_count']])

if __name__ == "__main__":
    main()
