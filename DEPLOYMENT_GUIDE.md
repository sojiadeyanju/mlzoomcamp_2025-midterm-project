# Crop Yield Prediction Web Service - Deployment Guide

This guide provides instructions for deploying the Crop Yield Prediction model as a web service using Docker, both locally and in the cloud.

## Project Structure

```
crop_yield_predictor/
├── train.py                 # Script to train the model
├── predict.py              # Flask API application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container configuration
├── models/                 # Directory for trained models (created at runtime)
│   ├── crop_yield_model.pkl
│   ├── feature_encoder.pkl
│   └── model_metrics.txt
└── DEPLOYMENT_GUIDE.md     # This file
```

## Prerequisites

Before deploying, ensure you have the following installed:
*   **Docker:** Download from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
*   **Docker Compose (optional):** For managing multi-container applications
*   **Git:** For version control

## Step 1: Prepare the Training Data

Place your cleaned crop yield dataset (`crop_yield_cleaned.csv`) in the project directory or specify its path via environment variables.

## Step 2: Train the Model Locally (Optional)

Before building the Docker image, you can train the model locally to verify everything works:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py
```

This will create the `models/` directory with the trained model and encoder files.

## Step 3: Build the Docker Image

Build the Docker image with the following command:

```bash
# Build the image
docker build -t crop-yield-predictor .

# Verify the image was created
docker images | grep crop-yield-predictor
```

## Step 4: Run the Container Locally

### Run the Container with Pre-trained Model

If you have already trained the model locally, copy the `models/` directory into the container:

```bash
# Run the container
docker run -d -p 5000:5000 crop-yield-predictor
```

## Step 5: Test the API Locally

Once the container is running, test the API endpoints:

### Health Check

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00.000000",
  "model_loaded": true
}
```

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Region": "North",
    "Soil_Type": "Sandy",
    "Crop": "Cotton",
    "Rainfall_mm": 897.08,
    "Temperature_Celsius": 27.68,
    "Fertilizer_Used": false,
    "Irrigation_Used": true,
    "Weather_Condition": "Cloudy",
    "Days_to_Harvest": 122
  }'
```

Expected response:
```json
{
  "predicted_yield": 6.5558,
  "unit": "tons_per_hectare",
  "timestamp": "2025-01-01T12:00:00.000000"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
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
      },
      {
        "Region": "South",
        "Soil_Type": "Clay",
        "Crop": "Rice",
        "Rainfall_mm": 992.67,
        "Temperature_Celsius": 18.03,
        "Fertilizer_Used": true,
        "Irrigation_Used": true,
        "Weather_Condition": "Rainy",
        "Days_to_Harvest": 140
      }
    ]
  }'
```

### Model Information

```bash
curl http://localhost:5000/info
```

## Step 6: Stop the Container

```bash
# Stop the running container
docker stop crop-yield-api

# Remove the container
docker rm crop-yield-api
```

## Cloud Deployment

### AWS Deployment (Amazon ECS)

1. **Create an AWS Account** and set up the AWS CLI.

2. **Push the Image to Amazon ECR:**

```bash
# Create an ECR repository
aws ecr create-repository --repository-name crop-yield-predictor

# Get the login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Tag the image
docker tag crop-yield-predictor:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/crop-yield-predictor:latest

# Push the image
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/crop-yield-predictor:latest
```

3. **Deploy to ECS:**
   - Create an ECS cluster
   - Create a task definition pointing to the ECR image
   - Create a service to run the task

### Google Cloud Deployment (Google Cloud Run)

1. **Set up Google Cloud CLI** and authenticate.

2. **Push the Image to Google Container Registry:**

```bash
# Set your project ID
export PROJECT_ID=$(gcloud config get-value project)

# Configure Docker authentication
gcloud auth configure-docker

# Tag the image
docker tag crop-yield-predictor:latest gcr.io/$PROJECT_ID/crop-yield-predictor:latest

# Push the image
docker push gcr.io/$PROJECT_ID/crop-yield-predictor:latest
```

3. **Deploy to Cloud Run:**

```bash
gcloud run deploy crop-yield-predictor \
  --image gcr.io/$PROJECT_ID/crop-yield-predictor:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000
```

### Azure Deployment (Azure Container Instances)

1. **Set up Azure CLI** and authenticate.

2. **Push the Image to Azure Container Registry:**

```bash
# Create a container registry
az acr create --resource-group myResourceGroup --name myregistry --sku Basic

# Get login credentials
az acr login --name myregistry

# Tag the image
docker tag crop-yield-predictor:latest myregistry.azurecr.io/crop-yield-predictor:latest

# Push the image
docker push myregistry.azurecr.io/crop-yield-predictor:latest
```

3. **Deploy to Container Instances:**

```bash
az container create \
  --resource-group myResourceGroup \
  --name crop-yield-predictor \
  --image myregistry.azurecr.io/crop-yield-predictor:latest \
  --registry-login-server myregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --ports 5000 \
  --environment-variables PORT=5000
```

## Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATA_FILE` | `/home/ubuntu/crop_yield_cleaned.csv` | Path to the training data CSV file. |
| `MODEL_FILE` | `models/crop_yield_model.pkl` | Path to save/load the trained model. |
| `ENCODER_FILE` | `models/feature_encoder.pkl` | Path to save/load the feature encoder. |
| `METRICS_FILE` | `models/model_metrics.txt` | Path to save model metrics. |
| `PORT` | `5000` | Port on which the Flask API listens. |

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Health check endpoint. Returns status and model load state. |
| `/predict` | POST | Single prediction endpoint. Accepts JSON with feature values. |
| `/batch-predict` | POST | Batch prediction endpoint. Accepts a list of feature records. |
| `/info` | GET | Model information endpoint. Returns model type and features. |

## Troubleshooting

### Issue: Container fails to start

**Solution:** Check the logs:
```bash
docker logs crop-yield-api
```

### Issue: Model files not found

**Solution:** Ensure the `models/` directory exists and contains the trained model files:
```bash
ls -la models/
```

### Issue: Prediction endpoint returns 500 error

**Solution:** Verify that all required fields are provided in the JSON request and that the data types are correct.

## Performance Optimization

For production deployments, consider the following optimizations:

1. **Use a Production WSGI Server:** The Dockerfile uses Gunicorn with 4 workers. Adjust the worker count based on your server's CPU cores.

2. **Enable Caching:** Implement caching for frequently requested predictions.

3. **Load Balancing:** Use a load balancer (e.g., AWS ELB, Google Cloud Load Balancer) to distribute traffic across multiple instances.

4. **Monitoring:** Set up monitoring and logging using tools like Prometheus, Grafana, or cloud-native monitoring services.

5. **Auto-scaling:** Configure auto-scaling policies to handle traffic spikes.

## Next Steps

1. Customize the API endpoints based on your specific requirements.
2. Add authentication and authorization if needed.
3. Implement comprehensive logging and monitoring.
4. Set up CI/CD pipelines for automated testing and deployment.
5. Consider adding additional features like model versioning and A/B testing.
