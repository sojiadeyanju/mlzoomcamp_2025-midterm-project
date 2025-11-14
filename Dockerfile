# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY predict.py .

# Copy the pre-trained models
COPY models/ ./models/

# Create models directory
# RUN mkdir -p models

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=predict.py
ENV PORT=5000
ENV MODEL_FILE=models/crop_yield_model.pkl
ENV ENCODER_FILE=models/feature_encoder.pkl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the Flask application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "predict:app"]
