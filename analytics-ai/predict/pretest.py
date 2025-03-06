import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv("bbps_fetch_txn_report.csv")

# Step 2: Prepare the data (group by date and blr_category)
df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])  # Ensure 'crtn_ts' is in datetime format
df['date'] = df['crtn_ts'].dt.date  # Extract date
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime64 format for Prophet compatibility

# Grouping transaction counts by date and category
transaction_counts = df.groupby(['date', 'blr_category']).size().reset_index(name='transactions')

# Rename columns for Prophet compatibility
transaction_counts = transaction_counts.rename(columns={'date': 'ds', 'transactions': 'y'})

# Step 3: Split data into training and test sets
train_size = int(len(transaction_counts) * 0.8)  # 80% for training, 20% for testing
train_data = transaction_counts[:train_size]
test_data = transaction_counts[train_size:]

print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Step 4: Initialize a dictionary to hold results for each category
category_forecasts = {}

# Step 5: Loop through each category to train and predict
for category in transaction_counts['blr_category'].unique():
    print(f"\nTraining model for category: {category}")
    
    # Filter data for the current category
    category_train_data = train_data[train_data['blr_category'] == category]
    category_test_data = test_data[test_data['blr_category'] == category]
    
    # Train Prophet model on the category's data
    model = Prophet(
        # changepoint_prior_scale=0.05,  # Tuning trend flexibility
        # seasonality_prior_scale=10,    # Tuning seasonality flexibility
        # yearly_seasonality=True,
        # weekly_seasonality=True,
        # daily_seasonality=True

    )
    model.fit(category_train_data[['ds', 'y']])

    # Make predictions for the test period
    future = model.make_future_dataframe(periods=len(category_test_data))
    forecast = model.predict(future)
    
    # Merge the forecast with the actual test data
    forecast_test = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]  # Predicted values
    category_test_data = category_test_data[['ds', 'y']]  # Actual values
    merged = pd.merge(category_test_data, forecast_test, on='ds', how='left')
    
    # Store the forecast in the dictionary
    category_forecasts[category] = merged

    # Step 6: Calculate evaluation metrics
    y_true = merged['y']  # Actual test values
    y_pred = merged['yhat']  # Predicted values
    
    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
    # mape = mean_absolute_percentage_error(y_true, y_pred)

    # print(f"Category: {category}")
    # print(f"MAE: {mae}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    # print(f"MAPE: {mape}")
    
    # Step 7: Visualize the predictions for each category
    plt.figure(figsize=(10, 6))
    plt.plot(merged['ds'], merged['y'], label='Actual')
    plt.plot(merged['ds'], merged['yhat'], label='Predicted', linestyle='--')
    plt.fill_between(merged['ds'], merged['yhat_lower'], merged['yhat_upper'], color='gray', alpha=0.2)
    plt.legend()
    plt.title(f"Actual vs. Predicted Transactions for Test Set - Category: {category}")
    plt.xlabel("Date")
    plt.ylabel("Transactions")
    plt.show()
    
    # Optionally: Save the forecast plot as a PNG file
    plot_filename = f"forecast_plot_{category}.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Forecast plot for {category} saved as {plot_filename}")

    # Optionally: Save the forecast results for each category to a CSV file
    forecast_filename = f"forecast_results_{category}.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_filename, index=False)
    print(f"Forecast results for {category} saved to {forecast_filename}")