#!/usr/bin/env python3
"""
train.py - Train the Linear Regression model for Crop Yield Prediction
This script loads the cleaned data, performs feature engineering, trains the model,
and saves it to a file using joblib for later use in the Flask API.
"""

import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Define paths
DATA_FILE = os.environ.get('DATA_FILE', 'crop_yield_cleaned.csv')
MODEL_FILE = os.environ.get('MODEL_FILE', 'models/crop_yield_model.pkl')
ENCODER_FILE = os.environ.get('ENCODER_FILE', 'models/feature_encoder.pkl')
METRICS_FILE = os.environ.get('METRICS_FILE', 'models/model_metrics.txt')

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_FILE) if os.path.dirname(MODEL_FILE) else '.', exist_ok=True)

def load_and_prepare_data(data_file):
    """Load the cleaned dataset and prepare features and target."""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    target_col = 'Yield_tons_per_hectare'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {target_col}")
    
    return X, y

def perform_feature_engineering(X):
    """Perform one-hot encoding on categorical features."""
    print("Performing feature engineering (One-Hot Encoding)...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Number of features after encoding: {X_encoded.shape[1]}")
    print(f"Feature names: {list(X_encoded.columns)}")
    
    return X_encoded

def train_model(X_train, y_train):
    """Train the Linear Regression model."""
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}

def save_model_and_encoder(model, X_encoded, model_file, encoder_file):
    """Save the trained model and feature encoder (column names) to files."""
    print(f"Saving model to {model_file}...")
    joblib.dump(model, model_file)
    
    print(f"Saving feature encoder to {encoder_file}...")
    joblib.dump(X_encoded.columns.tolist(), encoder_file)
    
    print("Model and encoder saved successfully.")

def save_metrics(metrics, metrics_file):
    """Save model metrics to a text file."""
    print(f"Saving metrics to {metrics_file}...")
    with open(metrics_file, 'w') as f:
        f.write("--- Model Metrics ---\n")
        for key, value in metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
    
    print("Metrics saved successfully.")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Crop Yield Prediction - Model Training Pipeline")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(DATA_FILE)
        
        # Perform feature engineering
        X_encoded = perform_feature_engineering(X)
        
        # Split data
        print("Splitting data into training (80%) and testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and encoder
        save_model_and_encoder(model, X_encoded, MODEL_FILE, ENCODER_FILE)
        
        # Save metrics
        save_metrics(metrics, METRICS_FILE)
        
        print("=" * 60)
        print("Training pipeline completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
