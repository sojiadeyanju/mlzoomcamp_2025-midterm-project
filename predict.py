#!/usr/bin/env python3
"""
predict.py - Flask API for Crop Yield Prediction
This script loads the trained Linear Regression model and serves predictions
via a REST API endpoint.
"""

import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Define paths
MODEL_FILE = os.environ.get('MODEL_FILE', 'models/crop_yield_model.pkl')
ENCODER_FILE = os.environ.get('ENCODER_FILE', 'models/feature_encoder.pkl')

# Global variables to store the model and encoder
model = None
feature_columns = None

def load_model():
    """Load the trained model and feature encoder from files."""
    global model, feature_columns
    
    # Check if files exist before trying to load
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file not found at {MODEL_FILE}", file=sys.stderr)
        return False
    if not os.path.exists(ENCODER_FILE):
        print(f"Error: Encoder file not found at {ENCODER_FILE}", file=sys.stderr)
        return False
        
    try:
        print(f"Loading model from {MODEL_FILE}...")
        model = joblib.load(MODEL_FILE)
        
        print(f"Loading feature encoder from {ENCODER_FILE}...")
        feature_columns = joblib.load(ENCODER_FILE)
        
        print("Model and encoder loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return False

@app.before_request
def before_request():
    """
    Ensure model is loaded before processing requests.
    Skip this check for the 'health_check' endpoint.
    """
    # Allow the health check to run even if the model is not loaded
    if request.endpoint == 'health_check':
        return

    if model is None or feature_columns is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_loaded_status = model is not None and feature_columns is not None
    
    if model_loaded_status:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_loaded_status
        }), 200
    else:
        # Service is "unhealthy" if model isn't loaded
        return jsonify({
            'status': 'unhealthy',
            'error': 'Model or feature encoder not loaded',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': model_loaded_status
        }), 503 # 503 Service Unavailable is appropriate here

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expected JSON input:
    {
        "Region": "North",
        "Soil_Type": "Sandy",
        "Crop": "Cotton",
        "Rainfall_mm": 897.08,
        "Temperature_Celsius": 27.68,
        "Fertilizer_Used": false,
        "Irrigation_Used": true,
        "Weather_Condition": "Cloudy",
        "Days_to_Harvest": 122
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
            'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all feature columns are present (add missing columns with 0)
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match the training data
        input_encoded = input_encoded[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        
        # Return prediction
        return jsonify({
            'predicted_yield': round(prediction, 4),
            'unit': 'tons_per_hectare',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    
    Expected JSON input:
    {
        "data": [
            {
                "Region": "North",
                "Soil_Type": "Sandy",
                ...
            },
            ...
        ]
    }
    """
    try:
        # Get JSON data from request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data array provided'}), 400
        
        data_list = request_data['data']
        
        if not isinstance(data_list, list):
            return jsonify({'error': 'Data must be a list'}), 400
        
        # Create DataFrame from list of records
        input_df = pd.DataFrame(data_list)
        
        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns
        input_encoded = input_encoded[feature_columns]
        
        # Make predictions
        predictions = model.predict(input_encoded)
        
        # Format results
        results = [
            {
                'predicted_yield': round(pred, 4),
                'unit': 'tons_per_hectare',
                'timestamp': datetime.utcnow().isoformat()
            }
            for pred in predictions
        ]
        
        return jsonify({'predictions': results}), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/info', methods=['GET'])
def info():
    """Return model information."""
    return jsonify({
        'model_type': 'Linear Regression',
        'features': feature_columns,
        'feature_count': len(feature_columns) if feature_columns else 0,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

# --- Load model on startup ---
# This code runs when the file is imported by Gunicorn or run directly.
if not load_model():
    print("FATAL: Failed to load model on startup. Check 'models' dir.", file=sys.stderr)
    # The app will continue to run, but /health will report 'model_loaded: false'
    # and all other endpoints will return 500 (due to @before_request).

if __name__ == '__main__':
    # This block only runs when script is executed directly (e.g., python predict.py)
    
    # Check if loading failed (for local development)
    if model is None or feature_columns is None:
        print("Model was not loaded successfully. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print(f"--- Running Flask app in DEBUG mode on http://0.0.0.0:{port} ---")
    # Run Flask app with debug=True for local development
    app.run(host='0.0.0.0', port=port, debug=True)