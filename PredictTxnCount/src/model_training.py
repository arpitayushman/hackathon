import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Regression Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def evaluate_model_performance(y_true, y_pred):
    """
    Comprehensive model performance evaluation
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_feature_importance(features, importances, model_name, prediction_type):
    """
    Visualize feature importances
    """
    plt.figure(figsize=(12, 6))
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.title(f'Feature Importance for {model_name} - {prediction_type.capitalize()} Prediction')
    sns.barplot(x='importance', y='feature', data=feature_imp.head(20))
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_{prediction_type}_feature_importance.png')
    plt.close()
    
    return feature_imp

def train_amount_model(X, y):
    """
    Train models for transaction amount prediction with comprehensive analysis
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and their parameter grids
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__solver': ['auto', 'svd', 'cholesky']
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__selection': ['cyclic', 'random']
            }
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.9]
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.3],
                'regressor__max_depth': [3, 5, 7]
            }
        }
    }
    
    # Results storage
    model_results = {}
    
    # Create a pipeline with scaling
    for name, model_info in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model_info['model'])
        ])
        
        # Grid Search
        grid_search = GridSearchCV(
            pipeline, 
            model_info['params'], 
            cv=5, 
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error'
        )
        
        # Fit Grid Search
        grid_search.fit(X_train, y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Evaluate performance
        performance = evaluate_model_performance(y_test, y_pred)
        
        # Feature importance for tree-based models
        if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
            feature_importances = best_model.named_steps['regressor'].feature_importances_
            feature_imp = plot_feature_importance(
                X.columns, 
                feature_importances, 
                name, 
                'amount'
            )
        
        # Store results
        model_results[name] = {
            'best_params': grid_search.best_params_,
            'performance': performance,
            'model': best_model
        }
    
    # Select best model based on lowest RMSE
    best_model_name = min(model_results, key=lambda x: model_results[x]['performance']['RMSE'])
    best_model = model_results[best_model_name]['model']
    
    # Save the best model
    joblib.dump(best_model, 'models/best_amount_prediction_model.joblib')
    
    print(f"Best Model for Amount Prediction: {best_model_name}")
    print("Model Performance:", model_results[best_model_name]['performance'])
    
    return model_results, best_model

def train_volume_model(X, y):
    """
    Train models for transaction volume prediction with comprehensive analysis
    Similar structure to train_amount_model
    """
    # Implementation similar to train_amount_model
    # Adapt the code for transaction volume prediction
    # Use the same grid search and evaluation approach
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and their parameter grids
    # (Same as train_amount_model, adjust as needed)
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    # Results storage
    model_results = {}
    
    # Create a pipeline with scaling
    for name, model_info in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model_info['model'])
        ])
        
        # Grid Search
        grid_search = GridSearchCV(
            pipeline, 
            model_info['params'], 
            cv=5, 
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error'
        )
        
        # Fit Grid Search
        grid_search.fit(X_train, y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Evaluate performance
        performance = evaluate_model_performance(y_test, y_pred)
        
        # Feature importance for tree-based models
        if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
            feature_importances = best_model.named_steps['regressor'].feature_importances_
            feature_imp = plot_feature_importance(
                X.columns, 
                feature_importances, 
                name, 
                'volume'
            )
        
        # Store results
        model_results[name] = {
            'best_params': grid_search.best_params_,
            'performance': performance,
            'model': best_model
        }
    
    # Select best model based on lowest RMSE
    best_model_name = min(model_results, key=lambda x: model_results[x]['performance']['RMSE'])
    best_model = model_results[best_model_name]['model']
    
    # Save the best model
    joblib.dump(best_model, 'models/best_volume_prediction_model.joblib')
    
    print(f"Best Model for Volume Prediction: {best_model_name}")
    print("Model Performance:", model_results[best_model_name]['performance'])
    
    return model_results, best_model

def optimize_hyperparameters(X, y, model_type='amount'):
    """
    Additional hyperparameter optimization function
    Demonstrates more advanced hyperparameter tuning
    """
    # Advanced grid search with more comprehensive parameter ranges
    if model_type == 'amount':
        # More extensive parameter grid for amount prediction
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        model = RandomForestRegressor(random_state=42)
    else:
        # Similar grid for volume prediction
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0]
        }
        model = XGBRegressor(random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    # Grid Search with more comprehensive scoring
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        refit='neg_mean_squared_error',
        n_jobs=-1  # Use all available cores
    )
    
    # Fit and print results
    grid_search.fit(X, y)
    
    print(f"Best Parameters for {model_type} prediction:")
    print(grid_search.best_params_)
    print("Best Cross-Validated Score:", -grid_search.best_score_)
    
    return grid_search.best_estimator_