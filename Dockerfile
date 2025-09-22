# Simple Dockerfile for first deployment
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn (simple, not gunicorn for first deployment)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]