# Docker Setup and Troubleshooting Guide

## Error Analysis

```
Unable to find image '7:latest' locally
docker: Error response from daemon: pull access denied for 7
```

This error typically occurs when:
1. The Docker command is malformed or incomplete
2. The image name is being parsed incorrectly
3. There's a syntax error in the docker command or Dockerfile

## Root Cause

The error suggests that Docker is interpreting something as just `7:latest`, which is not a valid image name. This commonly happens when:
- The image name is missing or incomplete in the command
- There's a typo or formatting issue in the docker build/run command
- The Dockerfile has an issue with the FROM statement

## Solution: Correct Docker Commands

### 1. Build the Docker Image

Navigate to the project directory and build the image with a proper name:

```bash
cd /home/ubuntu/crop_yield_predictor

# Build the image with a descriptive name
docker build -t crop-yield-predictor:latest .
```

**Expected output:**
```
Sending build context to Docker daemon  ...
Step 1/11 : FROM python:3.11-slim
...
Successfully built <image_id>
Successfully tagged crop-yield-predictor:latest
```

### 2. Verify the Image Was Built Successfully

```bash
docker images | grep crop-yield-predictor
```

You should see output like:
```
crop-yield-predictor   latest    <image_id>    <created_time>    <size>
```

### 3. Run the Container

Once the image is built, run it with:

```bash
# Basic run command
docker run -p 5000:5000 crop-yield-predictor:latest

# Run in detached mode (background)
docker run -d -p 5000:5000 --name crop-yield-api crop-yield-predictor:latest

# Run with volume mounting for models (recommended)
docker run -d \
  -p 5000:5000 \
  -v /home/ubuntu/crop_yield_predictor/models:/app/models \
  --name crop-yield-api \
  crop-yield-predictor:latest
```

### 4. Test the API

Once the container is running, test the endpoints:

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
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

### 5. View Container Logs

```bash
# View logs
docker logs crop-yield-api

# Follow logs in real-time
docker logs -f crop-yield-api
```

### 6. Stop and Remove Container

```bash
# Stop the container
docker stop crop-yield-api

# Remove the container
docker rm crop-yield-api

# Remove the image
docker rmi crop-yield-predictor:latest
```

## Common Docker Issues and Solutions

### Issue 1: "Cannot find module" or "ModuleNotFoundError"

**Solution:** Ensure all Python dependencies are installed. Check `requirements.txt` is in the project directory:

```bash
ls -la /home/ubuntu/crop_yield_predictor/requirements.txt
```

### Issue 2: Port Already in Use

**Solution:** Use a different port or stop the existing container:

```bash
# Find what's using port 5000
lsof -i :5000

# Kill the process or use a different port
docker run -p 8000:5000 crop-yield-predictor:latest
```

### Issue 3: Model Files Not Found

**Solution:** Ensure model files exist in the models directory:

```bash
ls -la /home/ubuntu/crop_yield_predictor/models/
```

Expected files:
- `crop_yield_model.pkl`
- `scaler.pkl`
- `model_metrics.json`
- `feature_names.json`

### Issue 4: Permission Denied

**Solution:** Run Docker with appropriate permissions:

```bash
# On Linux, you may need sudo
sudo docker build -t crop-yield-predictor:latest .
sudo docker run -p 5000:5000 crop-yield-predictor:latest

# Or add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
```

## Docker Compose Alternative

For easier management, create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  crop-yield-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - FLASK_APP=predict.py
      - PORT=5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

Then run with:

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Verification Checklist

Before running the container, verify:

- [ ] Dockerfile exists and is valid
- [ ] requirements.txt exists with all dependencies
- [ ] train.py and predict.py exist in the project directory
- [ ] models/ directory exists with trained model files
- [ ] Port 5000 is available (or use a different port)
- [ ] Docker daemon is running
- [ ] You have appropriate permissions to run Docker

## Next Steps

1. **Build the image** using the correct command above
2. **Run the container** with volume mounting for models
3. **Test the API** with sample predictions
4. **Monitor logs** for any errors
5. **Deploy to cloud** (AWS, GCP, Azure) using the built image

For cloud deployment, you would push the image to a container registry (Docker Hub, AWS ECR, etc.) and deploy from there.
