import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv("bbps_fetch_txn_report.csv")

# Step 2: Prepare the data (group by date and blr_category)
df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])  # Ensure 'crtn_ts' is in datetime format
df['date'] = df['crtn_ts'].dt.date  # Extract date
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime64 format for Prophet compatibility

# Grouping transaction counts by date and category
transaction_counts = df.groupby(['date', 'blr_category']).size().reset_index(name='transactions')

# Step 3: Loop through each unique category
categories = transaction_counts['blr_category'].unique()

for category in categories:
    # Filter data for the current category
    category_df = transaction_counts[transaction_counts['blr_category'] == category]
    
    # Rename columns for Prophet
    category_df = category_df.rename(columns={'date': 'ds', 'transactions': 'y'})
    
    # Drop rows with NaN values
    category_df = category_df.dropna(subset=['ds', 'y'])
    
    # Ensure there are enough data points
    if category_df.shape[0] < 2:
        print(f"Skipping {category} due to insufficient data.")
        continue  # Skip categories with less than 2 data points

    # Step 4: Initialize and train Prophet model
    model = Prophet()
    model.fit(category_df)
    print(f"Training data for category {category}:", category_df.head())  # Check the first few rows of the data
    
    # Step 5: Create future dates to predict (next 30 days)
    future = model.make_future_dataframe(periods=30)  # Only pass the dataframe, not 'periods' in the dataframe
    
    # Step 6: Make predictions
    forecast = model.predict(future)

    # Step 7: Save the forecast plot for this category as an image
    plt.figure(figsize=(10, 6))
    model.plot(forecast)
    plt.title(f'Forecast for {category}')
    
    # Save the plot as a PNG file
    plot_filename = f"{category}_forecast_plot.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Forecast plot for {category} saved as {plot_filename}")

    # Optionally: Save the forecast results for each category to a CSV file
    forecast_filename = f"{category}_forecast.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_filename, index=False)
    print(f"Forecast for {category} saved to {forecast_filename}")
