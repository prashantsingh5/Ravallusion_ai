# Simple Dockerfile for first deployment
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Copy local package source before installing requirements to support "-e ./revallusion_ai"
COPY revallusion_ai ./revallusion_ai

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn (simple, not gunicorn for first deployment)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]