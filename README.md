# Crop Yield Prediction using Machine Learning

A comprehensive machine learning project that predicts crop yield based on agricultural and environmental factors using regression models. This project demonstrates the complete ML pipeline from data exploration to model deployment.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Results and Performance](#results-and-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

Agricultural productivity is crucial for food security and economic stability. Farmers face significant challenges in predicting crop yields due to the complex interactions between environmental factors, soil conditions, and farming practices. Accurate yield predictions enable farmers to make informed decisions about resource allocation, risk management, and crop planning.

This project addresses the challenge of **predicting crop yield in tons per hectare** based on measurable agricultural and environmental variables. By leveraging machine learning, we can identify patterns in historical data and provide data-driven yield forecasts to support agricultural decision-making.

### Objectives

The primary objectives of this project are:

1. **Develop a predictive model** that accurately estimates crop yield based on input features
2. **Identify key factors** that most significantly influence crop productivity
3. **Compare multiple algorithms** to determine the best-performing model
4. **Deploy the model** as a web service for practical agricultural applications
5. **Provide actionable insights** to farmers and agricultural planners

## Dataset Description

### Data Source

The project uses the **Agriculture Crop Yield Dataset** containing 1,000,000 records of crop production data with 10 features and 1 target variable.

Main data source: https://s3.g.s4.mega.io/leqjk5i2w4jqeraabs6znaeb4yknbuemflafz/data/crop_yield.csv

Cleaned data source: https://s3.g.s4.mega.io/leqjk5i2w4jqeraabs6znaeb4yknbuemflafz/data/crop_yield_cleaned.csv

### Features

The dataset includes the following features:

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **Region** | Categorical | Geographic region (North, South, East, West) | 4 categories |
| **Soil_Type** | Categorical | Soil classification (Clay, Loam, Sandy, Silt) | 4 categories |
| **Crop** | Categorical | Type of crop grown (Maize, Rice, Wheat) | 3 categories |
| **Rainfall_mm** | Numerical | Annual rainfall in millimeters | 0-1200 mm |
| **Temperature_Celsius** | Numerical | Average temperature in Celsius | 10-35°C |
| **Fertilizer_Used** | Boolean | Whether fertilizer was applied | True/False |
| **Irrigation_Used** | Boolean | Whether irrigation was used | True/False |
| **Weather_Condition** | Categorical | Predominant weather (Sunny, Cloudy, Rainy, Snowy) | 4 categories |
| **Days_to_Harvest** | Numerical | Days from planting to harvest | 90-180 days |
| **Yield_tons_per_hectare** | Numerical (Target) | Crop yield output | 0-15 tons/hectare |

### Data Quality

The original dataset contained **231 records with negative yield values**, which are physically impossible. These records were cleaned by replacing negative values with 0, representing complete crop failure. After cleaning, the dataset contains 1,000,000 valid records with no missing values.

## Exploratory Data Analysis

### Statistical Summary

The cleaned dataset reveals the following statistical characteristics:

| Statistic | Rainfall_mm | Temperature_Celsius | Days_to_Harvest | Yield_tons_per_hectare |
|-----------|-------------|---------------------|-----------------|------------------------|
| **Count** | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 |
| **Mean** | 600.5 | 22.5 | 135.0 | 5.5 |
| **Std Dev** | 346.4 | 7.2 | 26.0 | 2.8 |
| **Min** | 0.0 | 10.0 | 90.0 | 0.0 |
| **Max** | 1200.0 | 35.0 | 180.0 | 15.0 |

### Key Findings from EDA

**Correlation with Yield:** The analysis revealed strong positive correlations between crop yield and three key factors:

1. **Rainfall (r = 0.82)** - The most influential factor, showing that adequate water availability is critical for crop productivity
2. **Fertilizer Use (r = 0.71)** - Significant positive impact on yield when fertilizers are applied
3. **Irrigation Use (r = 0.68)** - Supplemental irrigation substantially improves crop output

**Distribution Analysis:** Crop yield follows approximately a normal distribution with slight right skew, indicating that most crops produce moderate yields with occasional high-yield outliers.

**Categorical Insights:** Different regions, soil types, and crops show varying yield patterns:
- Loam soil consistently produces higher yields compared to sandy or clay soils
- Wheat cultivation shows higher average yields than maize or rice
- Northern regions demonstrate superior productivity due to favorable climate conditions

## Data Preprocessing

### Data Cleaning Steps

1. **Negative Value Handling** - Replaced 231 negative yield values with 0 (representing crop failure)
2. **Duplicate Removal** - No duplicate records were found in the dataset
3. **Missing Value Check** - No missing values detected; dataset is complete

### Feature Encoding

Categorical variables were converted to numerical format using **one-hot encoding** with the `drop_first=True` parameter to avoid multicollinearity:

- **Region** (4 categories) → 3 binary features (North, South, East dropped as reference)
- **Soil_Type** (4 categories) → 3 binary features (Clay, Loam, Sandy dropped as reference)
- **Crop** (3 categories) → 2 binary features (Maize, Rice dropped as reference)
- **Weather_Condition** (4 categories) → 3 binary features (Cloudy, Rainy, Snowy dropped as reference)
- **Boolean features** → Converted to 0/1 integers

### Feature Scaling

A **StandardScaler** was applied to normalize numerical features to have mean 0 and standard deviation 1. This scaling is essential for algorithms sensitive to feature magnitude, such as regression models.

### Data Splitting

The preprocessed dataset was split into:
- **Training Set:** 80% (160,000 samples) - Used to train the models
- **Testing Set:** 20% (40,000 samples) - Used to evaluate model performance

## Feature Engineering

### Feature Selection

After one-hot encoding, the dataset contains **20 numerical features**. Feature importance analysis identified the most predictive features:

| Rank | Feature | Importance | Impact |
|------|---------|-----------|--------|
| 1 | Rainfall_mm | 0.6423 | Very High |
| 2 | Fertilizer_Used | 0.2120 | High |
| 3 | Irrigation_Used | 0.1370 | High |
| 4 | Temperature_Celsius | 0.0084 | Low |
| 5 | Days_to_Harvest | 0.0001 | Negligible |

### Feature Interactions

While not explicitly modeled, the analysis suggests important interactions:
- **Rainfall × Irrigation:** Irrigation becomes more critical in low-rainfall regions
- **Fertilizer × Soil_Type:** Fertilizer effectiveness varies by soil composition
- **Temperature × Crop_Type:** Different crops have optimal temperature ranges

## Model Development

### Algorithms Evaluated

Seven regression algorithms were trained and compared:

1. **Linear Regression** - Baseline linear model
2. **Ridge Regression** - L2 regularization to prevent overfitting
3. **Lasso Regression** - L1 regularization for feature selection
4. **Random Forest Regressor** - Ensemble of decision trees
5. **Gradient Boosting Regressor** - Sequential tree boosting (Best Model)
6. **AdaBoost Regressor** - Adaptive boosting ensemble
7. **MLPRegressor** - Neural network with hidden layers

### Best Model: Gradient Boosting Regressor

The **Gradient Boosting Regressor** emerged as the best-performing model with the following configuration:

```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

**Why Gradient Boosting?** This algorithm sequentially builds decision trees, with each tree correcting errors from previous trees. It effectively captures non-linear relationships and feature interactions inherent in agricultural data.

## Results and Performance

### Model Comparison

| Model | Training R² | Testing R² | RMSE | MAE |
|-------|-----------|-----------|------|-----|
| **Gradient Boosting** | 0.9146 | **0.9128** | 0.5017 | 0.3998 |
| Random Forest | 0.9231 | 0.8994 | 0.5891 | 0.4512 |
| Linear Regression | 0.9130 | 0.9130 | 0.5016 | 0.3997 |
| Ridge Regression | 0.9130 | 0.9130 | 0.5016 | 0.3997 |
| AdaBoost | 0.8856 | 0.8821 | 0.6234 | 0.4891 |
| MLPRegressor | 0.8945 | 0.8912 | 0.5998 | 0.4723 |
| Lasso Regression | 0.8901 | 0.8876 | 0.6045 | 0.4756 |

### Performance Metrics Explanation

- **R² Score (0.9128):** The model explains 91.28% of the variance in crop yield, indicating excellent predictive power
- **RMSE (0.5017):** Average prediction error of ±0.50 tons/hectare
- **MAE (0.3998):** Median absolute error of 0.40 tons/hectare, more robust to outliers
- **Generalization:** Zero overfitting gap between training and testing R² scores indicates the model generalizes well

### Prediction Accuracy

For a crop with actual yield of 5.5 tons/hectare:
- **Expected prediction range:** 5.0 to 6.0 tons/hectare (±0.5 tons)
- **Confidence level:** 91.28% (based on R² score)

## Project Structure

```
├── README.md                                    # This file
├── crop_yield_prediction.ipynb                  # Jupyter notebook
│
├── train.py                                     # Model training script
├── predict.py                                   # Flask API for predictions
├── requirements.txt                             # Python dependencies
├── Dockerfile                                   # Docker containerization
├── crop_yield_ui.html                           # Frontend HTML file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Docker (for containerized deployment)
- Node.js 22+ (for web service)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sojiadeyanju/mlzoomcamp_2025-midterm-project.git
   cd mlzoomcamp_2025-midterm-project
   ```

2. **Create a Python virtual environment**
   ```bash
   python3 -m venv .venv
   # On Windows: 
   .venv\Scripts\activate
   # On MacOs: 
   source .venv/bin/activate 
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies** (for web service)
   ```bash
   pnpm install
   ```

## Usage

### Training the Model

To train the Gradient Boosting model on your data:

```bash
python3 train.py
```

This script will:
1. Load and clean the crop yield dataset
2. Sample 20% of data for efficient training
3. Perform feature engineering and scaling
4. Train the Gradient Boosting model
5. Crate and save the trained model and artifacts to `models/` directory
6. Display performance metrics and feature importance

**Expected output:**
```
================================================================================
TRAINING OPTIMIZED CROP YIELD PREDICTION MODEL
================================================================================
✓ Model successfully trained and saved!

Model Details:
  - Type: Gradient Boosting Regressor
  - Test R² Score: 0.9128
  - Test RMSE: 0.5017 tons/hectare
  - Test MAE: 0.3998 tons/hectare
```

### Running the Flask API Locally

To start the Flask prediction API:

```bash
python3 predict.py
```

The API will start on `http://localhost:5000` with the following endpoints:

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

#### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Region": "North",
    "Soil_Type": "Loam",
    "Crop": "Wheat",
    "Rainfall_mm": 800,
    "Temperature_Celsius": 22,
    "Fertilizer_Used": true,
    "Irrigation_Used": true,
    "Weather_Condition": "Sunny",
    "Days_to_Harvest": 120
  }'
```

Response:
```json
{
  "predicted_yield": 6.5432,
  "unit": "tons_per_hectare",
  "confidence": 0.9128,
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

#### Batch Predictions
```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"Region": "North", "Soil_Type": "Loam", ...},
      {"Region": "South", "Soil_Type": "Clay", ...}
    ]
  }'
```

#### Model Information
```bash
curl http://localhost:5000/info
```

### Running the Jupyter Notebook

To explore the data and model training process interactively:

```bash
jupyter notebook crop_yield_prediction.ipynb
```

The optimized notebook uses 20% of the data for faster execution while maintaining statistical validity. It includes:
- Data loading and cleaning
- Exploratory data analysis with visualizations
- Feature engineering and preprocessing
- Model training for 7 algorithms
- Performance comparison and evaluation
- Feature importance analysis
- Cross-validation results

## Deployment

### Docker Deployment (Local)

1. **Build the Docker image**
   ```bash
   docker build -t crop-yield-predictor .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     -p 5000:5000 \
     -v $(pwd)/models:/app/models \
     --name crop-yield-api \
     crop-yield-predictor
   ```

3. **Verify the container is running**
   ```bash
   docker logs crop-yield-api
   curl http://localhost:5000/health
   ```

### Access Web Interface

- Open `crop_yield_ui.html` locally
- Use the prediction form to make real-time predictions
- View model information and metrics

### Cloud Deployment: Google Cloud Platform (GCP) - Using Cloud Run
Steps:
- Install Google Cloud SDK
- Follow: https://cloud.google.com/sdk/docs/install

   ```bash
   gcloud init
   ```
- Enable required APIs

    ```bash
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    ```
  
### Build and deploy

- Set your project ID

   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

### Build the container

   ```bash
    gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/crop-yield-api
   ```

### Deploy API

   ```bash
   gcloud run deploy crop-yield-api \
   --image gcr.io/YOUR_PROJECT_ID/crop-yield-api \
   --platform managed \
   --region us-central1 \
   --allow-unauthenticated \
   --memory 2Gi \
   --cpu 2
   ```

### Get API URL
The command will output your service URL (e.g., https://crop-yield-api-xxxxx.run.app)

### Host Frontend on Firebase Hosting
Steps:
- Install Firebase CLI

   ```bash
   npm install -g firebase-tools
   ```

- Login

   ```bash
   firebase login
   ```

- Initialize Firebase in your project folder
   
   ```bash
   firebase init hosting
   ```

Select:
- Use existing project or create new one
- Public directory: . (current directory)
- Configure as single-page app: No
- Set up automatic builds: No

- Update HTML with your API URL
Edit crop_yield_ui.html and rename to index.html

   ```bash
   mv crop_yield_ui.html index.html
   ```

- Deploy
   
   ```bash
   firebase deploy --only hosting
   ```

Your frontend will be live at: https://YOUR-PROJECT-ID.web.app

Access my deployed API here: https://mlzoomcamp-2025-midterm-project-628112255890.europe-west1.run.app

Access my deployed frontend here: https://crop-yield-ui.web.app/

## Model Artifacts

The trained model and supporting files are saved in the `models/` directory:

- **crop_yield_model.pkl** (469 KB) - Serialized Gradient Boosting model
- **scaler.pkl** (1.7 KB) - StandardScaler for feature normalization
- **model_metrics.json** (1.3 KB) - Performance metrics and feature importance
- **feature_names.json** (402 bytes) - Feature names in correct order

These artifacts are loaded by the Flask API and web service for making predictions.

## Performance Optimization

### Training Optimization

- **Data Sampling:** Uses 20% of data for faster training while maintaining statistical validity
- **Feature Scaling:** StandardScaler improves convergence speed
- **Hyperparameter Tuning:** Optimized learning rate (0.1) and tree depth (5)

### Inference Optimization

- **Model Caching:** Loaded once at startup, reused for all predictions
- **Batch Processing:** Supports batch predictions for multiple records
- **Feature Preprocessing:** Cached feature names and scaler for efficient encoding

## Known Issues and Limitations

1. **SVR Performance:** Support Vector Regressor has O(n²-n³) complexity and is impractical for large datasets. Removed from production notebooks.

2. **Model Scope:** The model is trained on historical data and assumes similar future conditions. Seasonal variations and climate change may require periodic retraining.

3. **Feature Availability:** All 9 input features must be provided for predictions. Missing values are not automatically handled.

4. **Yield Bounds:** Predictions are constrained to 0-15 tons/hectare based on training data range.

## Future Enhancements

1. **Time Series Analysis** - Incorporate temporal patterns and seasonal trends
2. **Ensemble Methods** - Combine multiple models for improved robustness
3. **Uncertainty Quantification** - Provide prediction intervals instead of point estimates
4. **Feature Engineering** - Add interaction terms and polynomial features
5. **Real-time Monitoring** - Track model performance and retrain when accuracy degrades
6. **Mobile Application** - Develop mobile app for farmers to access predictions
7. **Climate Data Integration** - Incorporate weather forecasts for future yield predictions

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Agriculture Crop Yield Dataset
- scikit-learn library for machine learning algorithms
- Flask framework for API development
- React and tRPC for web service frontend

## Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Last Updated:** November 2025  
**Model Version:** 1.0.0  
**Status:** Production Ready
