#!/usr/bin/env python3
"""
predict.py - Flask API for Crop Yield Prediction
This script loads the trained Gradient Boosting model and serves predictions
via a REST API endpoint.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Define paths
MODEL_FILE = os.environ.get('MODEL_FILE', 'models/crop_yield_model.pkl')
SCALER_FILE = os.environ.get('SCALER_FILE', 'models/scaler.pkl')
METRICS_FILE = os.environ.get('METRICS_FILE', 'models/model_metrics.json')
FEATURE_NAMES_FILE = os.environ.get('FEATURE_NAMES_FILE', 'models/feature_names.json')

# Global variables to store the model and related objects
model = None
scaler = None
feature_names = None
model_metrics = None

def load_model():
    """Load the trained model and related files."""
    global model, scaler, feature_names, model_metrics
    
    try:
        print("\n" + "="*80)
        print("LOADING MODEL ARTIFACTS")
        print("="*80)
        
        # Load the model
        if os.path.exists(MODEL_FILE):
            print(f"Loading model from {MODEL_FILE}...")
            model = joblib.load(MODEL_FILE)
            print(f"✓ Model loaded successfully: {type(model).__name__}")
        else:
            print(f"✗ ERROR: Model file not found at {MODEL_FILE}")
            print(f"  Current working directory: {os.getcwd()}")
            print(f"  Files in models/: {os.listdir('models/') if os.path.exists('models/') else 'Directory does not exist'}")
            return False
        
        # Load the scaler
        if os.path.exists(SCALER_FILE):
            print(f"Loading scaler from {SCALER_FILE}...")
            scaler = joblib.load(SCALER_FILE)
            print(f"✓ Scaler loaded successfully: {type(scaler).__name__}")
        else:
            print(f"✗ WARNING: Scaler file not found at {SCALER_FILE}")
            scaler = None
        
        # Load feature names
        if os.path.exists(FEATURE_NAMES_FILE):
            print(f"Loading feature names from {FEATURE_NAMES_FILE}...")
            with open(FEATURE_NAMES_FILE, 'r') as f:
                feature_names = json.load(f)
            print(f"✓ Feature names loaded successfully ({len(feature_names)} features)")
        else:
            print(f"✗ WARNING: Feature names file not found at {FEATURE_NAMES_FILE}")
            feature_names = None
        
        # Load model metrics - check both .json and .txt extensions
        metrics_file_to_use = METRICS_FILE
        if not os.path.exists(METRICS_FILE):
            # Try .txt extension if .json doesn't exist
            alt_metrics_file = METRICS_FILE.replace('.json', '.txt')
            if os.path.exists(alt_metrics_file):
                metrics_file_to_use = alt_metrics_file
        
        if os.path.exists(metrics_file_to_use):
            print(f"Loading model metrics from {metrics_file_to_use}...")
            with open(metrics_file_to_use, 'r') as f:
                model_metrics = json.load(f)
            print(f"✓ Model metrics loaded successfully")
            print(f"  - Model Type: {model_metrics.get('model_type', 'Unknown')}")
            print(f"  - Test R² Score: {model_metrics.get('testing_metrics', {}).get('r2_score', 'N/A')}")
        else:
            print(f"✗ WARNING: Model metrics file not found at {METRICS_FILE}")
            model_metrics = None
        
        print("="*80)
        print("✓ ALL MODEL ARTIFACTS LOADED SUCCESSFULLY!")
        print("="*80 + "\n")
        return True
    
    except Exception as e:
        print(f"\n✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load model at module level (when Gunicorn imports this file)
print("Initializing application...")
if not load_model():
    print("WARNING: Failed to load model on startup")

@app.before_request
def before_request():
    """Ensure model is loaded before processing requests."""
    if model is None or feature_names is None:
        return jsonify({
            'error': 'Model or feature encoder not loaded',
            'model_loaded': model is not None,
            'feature_names_loaded': feature_names is not None,
            'status': 'unhealthy'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if (model is not None and feature_names is not None) else 'unhealthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names_loaded': feature_names is not None,
        'model_type': model_metrics.get('model_type') if model_metrics else None,
        'test_r2_score': model_metrics.get('testing_metrics', {}).get('r2_score') if model_metrics else None
    }), 200 if (model is not None and feature_names is not None) else 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expected JSON input:
    {
        "Region": "North",
        "Soil_Type": "Loam",
        "Crop": "Wheat",
        "Rainfall_mm": 800,
        "Temperature_Celsius": 22,
        "Fertilizer_Used": true,
        "Irrigation_Used": true,
        "Weather_Condition": "Sunny",
        "Days_to_Harvest": 120
    }
    
    Returns:
    {
        "predicted_yield": 6.55,
        "unit": "tons_per_hectare",
        "timestamp": "2025-01-01T12:00:00.000000"
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
        if feature_names:
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match the training data
            input_encoded = input_encoded[feature_names]
        
        # Scale features if scaler is available
        if scaler is not None:
            input_scaled = scaler.transform(input_encoded)
            input_encoded = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        
        # Ensure prediction is within reasonable bounds
        prediction = max(0, min(15, float(prediction)))
        
        # Return prediction
        return jsonify({
            'predicted_yield': round(prediction, 4),
            'unit': 'tons_per_hectare',
            'timestamp': datetime.utcnow().isoformat(),
            'confidence': model_metrics.get('testing_metrics', {}).get('r2_score') if model_metrics else None
        }), 200
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
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
                "Soil_Type": "Loam",
                ...
            },
            ...
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"predicted_yield": 6.55, "timestamp": "..."},
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
        if feature_names:
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns
            input_encoded = input_encoded[feature_names]
        
        # Scale features if scaler is available
        if scaler is not None:
            input_scaled = scaler.transform(input_encoded)
            input_encoded = pd.DataFrame(input_scaled, columns=feature_names)
        
        # Make predictions
        predictions = model.predict(input_encoded)
        
        # Format results
        results = [
            {
                'predicted_yield': round(max(0, min(15, float(pred))), 4),
                'unit': 'tons_per_hectare',
                'timestamp': datetime.utcnow().isoformat()
            }
            for pred in predictions
        ]
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        }), 200
    
    except Exception as e:
        print(f"Batch prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500

@app.route('/info', methods=['GET'])
def info():
    """Return model information."""
    return jsonify({
        'model_type': model_metrics.get('model_type') if model_metrics else 'Unknown',
        'features': feature_names,
        'feature_count': len(feature_names) if feature_names else 0,
        'metrics': model_metrics.get('testing_metrics') if model_metrics else None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        'name': 'Crop Yield Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'GET /info': 'Model information',
            'POST /predict': 'Single prediction',
            'POST /batch-predict': 'Batch predictions'
        },
        'status': 'running' if (model is not None and feature_names is not None) else 'unhealthy'
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CROP YIELD PREDICTION API - STARTUP")
    print("="*80)
    print(f"Working directory: {os.getcwd()}")
    print(f"Model file: {MODEL_FILE}")
    print(f"Scaler file: {SCALER_FILE}")
    print(f"Feature names file: {FEATURE_NAMES_FILE}")
    print(f"Metrics file: {METRICS_FILE}")
    
    # Load model on startup (if not already loaded)
    if model is None:
        if not load_model():
            print("\n" + "="*80)
            print("✗ ERROR: Failed to load model. Exiting.")
            print("="*80)
            sys.exit(1)
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*80)
    print(f"Starting Flask API on port {port}...")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=False)