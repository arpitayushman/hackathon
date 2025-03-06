# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from fastapi import FastAPI
# import uvicorn

# # Initialize FastAPI app
# app = FastAPI()

# # Step 1: Load the data
# df = pd.read_csv("bbps_fetch_txn_report.csv")

# # Step 2: Prepare the data
# df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])
# df['date'] = df['crtn_ts'].dt.date
# df['date'] = pd.to_datetime(df['date'])

# # Grouping transaction counts by date and category
# transaction_counts = df.groupby(['date', 'blr_category']).size().reset_index(name='transactions')

# # Store trained models in a dictionary
# models = {}

# # Step 3: Train Prophet models for each category
# categories = transaction_counts['blr_category'].unique()
# for category in categories:
#     category_df = transaction_counts[transaction_counts['blr_category'] == category]
    
#     # Rename columns for Prophet
#     category_df = category_df.rename(columns={'date': 'ds', 'transactions': 'y'})
    
#     # Drop NaN values
#     category_df = category_df.dropna(subset=['ds', 'y'])
    
#     # Ensure sufficient data points
#     if category_df.shape[0] < 2:
#         print(f"Skipping {category} due to insufficient data.")
#         continue  

#     # Train Prophet model
#     model = Prophet()
#     model.fit(category_df)

#     # Store the trained model
#     models[category] = model

# @app.get("/predict/{category}")
# def predict(category: str, periods: int = 30):
#     """
#     API endpoint to get predictions for a specific category.
#     :param category: The transaction category to predict.
#     :param periods: The number of future days to predict (default: 30).
#     :return: JSON response with predicted values.
#     """
#     if category not in models:
#         return {"error": "Category not found or insufficient data."}

#     model = models[category]
    
#     # Create future dates
#     future = model.make_future_dataframe(periods=periods)
    
#     # Get predictions
#     forecast = model.predict(future)
    
#     # Convert to dictionary format
#     predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')
    
#     return {"category": category, "predictions": predictions}

# # Run the FastAPI server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from fastapi import FastAPI
# import uvicorn
# import io
# import base64
# from fastapi.responses import JSONResponse

# # Initialize FastAPI app
# app = FastAPI()

# # Step 1: Load the data
# df = pd.read_csv("bbps_fetch_txn_report.csv")

# # Step 2: Prepare the data
# df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])
# df['date'] = df['crtn_ts'].dt.date
# df['date'] = pd.to_datetime(df['date'])

# # Grouping transaction counts by date and category
# transaction_counts = df.groupby(['date', 'blr_category']).size().reset_index(name='transactions')

# # Store trained models in a dictionary
# models = {}

# # Step 3: Train Prophet models for each category
# categories = transaction_counts['blr_category'].unique()
# for category in categories:
#     category_df = transaction_counts[transaction_counts['blr_category'] == category]
    
#     # Rename columns for Prophet
#     category_df = category_df.rename(columns={'date': 'ds', 'transactions': 'y'})
    
#     # Drop NaN values
#     category_df = category_df.dropna(subset=['ds', 'y'])
    
#     # Ensure sufficient data points
#     if category_df.shape[0] < 2:
#         print(f"Skipping {category} due to insufficient data.")
#         continue  

#     # Train Prophet model
#     model = Prophet()
#     model.fit(category_df)

#     # Store the trained model
#     models[category] = model

# def generate_plot(model, forecast):
#     """
#     Generate a base64-encoded image of the prediction plot.
#     """
#     plt.figure(figsize=(10, 6))
#     model.plot(forecast)
#     plt.title("Prediction Forecast")

#     # Save the plot to a BytesIO buffer
#     img_buf = io.BytesIO()
#     plt.savefig(img_buf, format='png')
#     plt.close()
    
#     # Encode image to base64 string
#     img_buf.seek(0)
#     encoded_img = base64.b64encode(img_buf.getvalue()).decode("utf-8")
#     return encoded_img

# @app.get("/predict/{category}")
# def predict(category: str, periods: int = 30):
#     """
#     API endpoint to get predictions for a specific category.
#     :param category: The transaction category to predict.
#     :param periods: The number of future days to predict (default: 30).
#     :return: JSON response with predicted values and graph.
#     """
#     if category not in models:
#         return JSONResponse(content={"error": "Category not found or insufficient data."}, status_code=404)

#     model = models[category]
    
#     # Create future dates
#     future = model.make_future_dataframe(periods=periods)
    
#     # Get predictions
#     forecast = model.predict(future)
    
#     # Generate the forecast plot
#     plot_image = generate_plot(model, forecast)
    
#     # Convert forecast data to JSON
#     predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')

#     return {
#         "category": category,
#         "predictions": predictions,
#         "plot_image": f"data:image/png;base64,{plot_image}"  # Base64-encoded image
#     }

# # Run the FastAPI server
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn
import io
import base64
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Load the data
df = pd.read_csv("bbps_fetch_txn_report.csv")

# Prepare the data
df['crtn_ts'] = pd.to_datetime(df['crtn_ts'])
df['date'] = df['crtn_ts'].dt.date
df['date'] = pd.to_datetime(df['date'])

# Grouping transaction counts by date and category
transaction_counts = df.groupby(['date', 'blr_category']).size().reset_index(name='transactions')

# Store trained models in a dictionary
models = {}

# Train Prophet models for each category
categories = transaction_counts['blr_category'].unique()
for category in categories:
    category_df = transaction_counts[transaction_counts['blr_category'] == category]
    
    # Rename columns for Prophet
    category_df = category_df.rename(columns={'date': 'ds', 'transactions': 'y'})
    
    # Drop NaN values
    category_df = category_df.dropna(subset=['ds', 'y'])
    
    # Ensure sufficient data points
    if category_df.shape[0] < 2:
        print(f"Skipping {category} due to insufficient data.")
        continue  

    # Train Prophet model
    model = Prophet()
    model.fit(category_df)

    # Store the trained model
    models[category] = model

def generate_improved_plot(model, forecast, category):
    """
    Generates a visually improved forecast plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data
    model.history.plot(x='ds', y='y', ax=ax, label="Actual Transactions", linestyle='-', marker='o', color='blue')

    # Plot predicted data
    ax.plot(forecast['ds'], forecast['yhat'], label="Predicted Transactions", color='green', linestyle='dashed')

    # Fill confidence interval
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='green', alpha=0.2, label="Confidence Interval")

    # Improve labels and title
    ax.set_title(f"Transaction Forecast for {category}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Transaction Count", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Convert the plot to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    
    img_buf.seek(0)
    encoded_img = base64.b64encode(img_buf.getvalue()).decode("utf-8")
    return encoded_img

@app.get("/predict/{category}")
def predict(category: str, periods: int = 30):
    """
    API endpoint to get predictions for a specific category.
    :param category: The transaction category to predict.
    :param periods: The number of future days to predict (default: 30).
    :return: JSON response with predicted values and improved graph.
    """
    if category not in models:
        return JSONResponse(content={"error": "Category not found or insufficient data."}, status_code=404)

    model = models[category]
    
    # Create future dates
    future = model.make_future_dataframe(periods=periods)
    
    # Get predictions
    forecast = model.predict(future)
    
    # Generate improved forecast plot
    plot_image = generate_improved_plot(model, forecast, category)
    
    # Convert forecast data to JSON
    predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')

    return {
        "category": category,
        "predictions": predictions,
        "plot_image": f"data:image/png;base64,{plot_image}"  # Base64-encoded image
    }

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

