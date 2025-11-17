#!/usr/bin/env python3
"""
train.py - Train the Gradient Boosting model for Crop Yield Prediction
This script loads the cleaned data, performs feature engineering, trains the model,
and saves it to a file using joblib for later use in the Flask API.
"""

import sys
import os
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Define paths
DATA_FILE = os.environ.get('DATA_FILE', 'https://s3.g.s4.mega.io/leqjk5i2w4jqeraabs6znaeb4yknbuemflafz/data/crop_yield_cleaned.csv')
MODEL_FILE = os.environ.get('MODEL_FILE', 'models/crop_yield_model.pkl')
ENCODER_FILE = os.environ.get('ENCODER_FILE', 'models/feature_encoder.pkl')
METRICS_FILE = os.environ.get('METRICS_FILE', 'models/model_metrics.txt')
SCALER_FILE = os.environ.get('SCALER_FILE', 'models/scaler.pkl')
FEATURE_NAMES_FILE = os.environ.get('FEATURE_NAMES_FILE', 'models/feature_names.json')

# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_FILE) if os.path.dirname(MODEL_FILE) else '.', exist_ok=True)

def load_and_prepare_data(data_file, sample_fraction=0.2):
    """Load the cleaned dataset and prepare features and target."""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # Sample the data for faster training
    print(f"Sampling {sample_fraction*100:.0f}% of data for training...")
    df = df.sample(frac=sample_fraction, random_state=42)
    
    target_col = 'Yield_tons_per_hectare'
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {target_col}")
    print(f"Features: {X.shape[1]}")
    
    return X, y

def perform_feature_engineering(X):
    """Perform one-hot encoding on categorical features."""
    print("Performing feature engineering (One-Hot Encoding)...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print(f"Number of features after encoding: {X_encoded.shape[1]}")
    print(f"Feature names: {list(X_encoded.columns)}")
    
    return X_encoded

def scale_features(X_train, X_test):
    """Scale numerical features using StandardScaler."""
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("Features scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """Train the Gradient Boosting Regressor model."""
    print("Training Gradient Boosting Regressor model...")
    print("  - n_estimators: 100")
    print("  - learning_rate: 0.1")
    print("  - max_depth: 5")
    print("  - random_state: 42")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the model on both training and test sets."""
    print("\nEvaluating model...")
    
    # Training metrics
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Testing metrics
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n--- Training Metrics ---")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MAE: {train_mae:.4f} tons/hectare")
    print(f"  RMSE: {train_rmse:.4f} tons/hectare")
    
    print("\n--- Testing Metrics ---")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.4f} tons/hectare")
    print(f"  RMSE: {test_rmse:.4f} tons/hectare")
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    metrics = {
        'model_type': 'Gradient Boosting Regressor',
        'training_metrics': {
            'r2_score': float(train_r2),
            'mae': float(train_mae),
            'rmse': float(train_rmse)
        },
        'testing_metrics': {
            'r2_score': float(test_r2),
            'mae': float(test_mae),
            'rmse': float(test_rmse)
        },
        'feature_importance': feature_importance.tolist()
    }
    
    return metrics

def save_model_and_artifacts(model, scaler, X_encoded, model_file, scaler_file, feature_names_file):
    """Save the trained model, scaler, and feature names."""
    print(f"\nSaving model to {model_file}...")
    joblib.dump(model, model_file)
    print(f"✓ Model saved ({os.path.getsize(model_file) / 1024:.1f} KB)")
    
    print(f"Saving scaler to {scaler_file}...")
    joblib.dump(scaler, scaler_file)
    print(f"✓ Scaler saved ({os.path.getsize(scaler_file) / 1024:.1f} KB)")
    
    print(f"Saving feature names to {feature_names_file}...")
    with open(feature_names_file, 'w') as f:
        json.dump(X_encoded.columns.tolist(), f, indent=2)
    print(f"✓ Feature names saved ({os.path.getsize(feature_names_file) / 1024:.1f} KB)")

def save_metrics(metrics, metrics_file):
    """Save model metrics to a JSON file."""
    print(f"Saving metrics to {metrics_file}...")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved ({os.path.getsize(metrics_file) / 1024:.1f} KB)")

def main():
    """Main training pipeline."""
    print("=" * 80)
    print("CROP YIELD PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data(DATA_FILE, sample_fraction=0.2)
        
        # Perform feature engineering
        X_encoded = perform_feature_engineering(X)
        
        # Split data
        print("\nSplitting data into training (80%) and testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Testing samples: {len(X_test):,}")
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save model and artifacts
        save_model_and_artifacts(model, scaler, X_encoded, MODEL_FILE, SCALER_FILE, FEATURE_NAMES_FILE)
        
        # Save metrics
        save_metrics(metrics, METRICS_FILE)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nModel Details:")
        print(f"  - Type: {metrics['model_type']}")
        print(f"  - Test R² Score: {metrics['testing_metrics']['r2_score']:.4f}")
        print(f"  - Test RMSE: {metrics['testing_metrics']['rmse']:.4f} tons/hectare")
        print(f"  - Test MAE: {metrics['testing_metrics']['mae']:.4f} tons/hectare")
        print(f"\nModel files saved:")
        print(f"  - {MODEL_FILE}")
        print(f"  - {SCALER_FILE}")
        print(f"  - {FEATURE_NAMES_FILE}")
        print(f"  - {METRICS_FILE}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
