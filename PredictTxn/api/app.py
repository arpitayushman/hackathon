from flask import Flask, request, jsonify, send_file
import subprocess
import os
import pandas as pd
import joblib
from flask_cors import CORS
import zipfile
import tempfile
import re
from config import VALID_CATEGORIES, BASE_DIR

app = Flask(__name__)
CORS(app)
# Load models
def load_models(category):
    models_dir = os.path.join(BASE_DIR, 'models')
    
    try:
        # Load appropriate models based on category
        if category == 'Loan':
            amount_model = joblib.load(os.path.join(models_dir, 'amount_model.joblib'))
            volume_model = joblib.load(os.path.join(models_dir, 'volume_model.joblib'))
        elif category == 'Utility':
            amount_model = joblib.load(os.path.join(models_dir, 'amount_model.joblib'))
            volume_model = joblib.load(os.path.join(models_dir, 'volume_model.joblib'))
        else:
            # For other categories, you might need specific models or a generic one
            amount_model = joblib.load(os.path.join(models_dir, 'amount_model.joblib'))
            volume_model = joblib.load(os.path.join(models_dir, 'volume_model.joblib'))
        
        return amount_model, volume_model
    except FileNotFoundError:
        raise ValueError(f"Model files for category '{category}' not found")

@app.route('/train', methods=['POST'])
def train_model():
    """API endpoint to trigger model training"""
    try:
        # Execute the training script
        result = subprocess.run(['python', 'main.py', '--train'], 
                               capture_output=True, text=True, check=True)
        
        return jsonify({
            'status': 'success',
            'message': 'Model training completed',
            'details': result.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            'status': 'error',
            'message': 'Model training failed',
            'details': e.stderr
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions and return encoded files in response"""
    # Get parameters from request
    data = request.json
    category = data.get('category')
    date = data.get('date')
    
    # Validate input parameters
    if not category:
        return jsonify({
            'status': 'error',
            'message': 'Category parameter is required'
        }), 400
    
    if category not in VALID_CATEGORIES:
        return jsonify({
            'status': 'error',
            'message': f"Invalid category. Valid categories are: {', '.join(VALID_CATEGORIES)}"
        }), 400
    
    if not date:
        return jsonify({
            'status': 'error',
            'message': 'Date parameter is required'
        }), 400
    
    # Validate date format
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        return jsonify({
            'status': 'error',
            'message': 'Date should be in YYYY-MM-DD format'
        }), 400
    
    try:
        # Execute the prediction script
        result = subprocess.run(
            ['python', 'main.py', '--predict', '--category', category, '--date', date],
            capture_output=True, text=True, check=True
        )
        
        # Format date for filename (remove hyphens if your script does)
        formatted_date = date.replace('-', '')
        
        # Initialize response data
        response_data = {
            'status': 'success',
            'message': 'Prediction completed successfully',
            'files': {}
        }
        
        # Process CSV file
        csv_path = os.path.join(BASE_DIR, 'results', f"{category}_{formatted_date}_predictions.csv")
        if os.path.exists(csv_path):
            # Read CSV data and convert to a list of dictionaries
            try:
                df = pd.read_csv(csv_path)
                response_data['files']['csv'] = {
                    'filename': f"{category}_{formatted_date}_predictions.csv",
                    'data': df.to_dict(orient='records')
                }
            except Exception as e:
                response_data['files']['csv'] = {
                    'error': f"Failed to process CSV: {str(e)}"
                }
        else:
            return jsonify({
                'status': 'error',
                'message': f"CSV file for {category} and date {date} not found"
            }), 404
        
        # Process PNG files
        png_files = [
            {
                'path': os.path.join(BASE_DIR, 'results', f"{category}_predictions.png"),
                'name': f"{category}_predictions.png"
            },
            {
                'path': os.path.join(BASE_DIR, 'results', f"{category}_total_amount.png"),
                'name': f"{category}_total_amount.png"
            }
        ]
        
        response_data['files']['images'] = []
        
        import base64
        for png_file in png_files:
            if os.path.exists(png_file['path']):
                try:
                    with open(png_file['path'], 'rb') as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                        response_data['files']['images'].append({
                            'filename': png_file['name'],
                            'data': encoded_image,
                            'mime_type': 'image/png'
                        })
                except Exception as e:
                    response_data['files']['images'].append({
                        'filename': png_file['name'],
                        'error': f"Failed to encode image: {str(e)}"
                    })
            else:
                response_data['files']['images'].append({
                    'filename': png_file['name'],
                    'error': 'File not found'
                })
        
        return jsonify(response_data)
            
    except subprocess.CalledProcessError as e:
        return jsonify({
            'status': 'error',
            'message': 'Prediction failed',
            'details': e.stderr
        }), 500
@app.route('/get_categories', methods=['GET'])
def get_categories():
    """API endpoint to return available biller categories"""
    return jsonify({
        'status': 'success',
        'categories': VALID_CATEGORIES
    })