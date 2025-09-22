# Simple Dockerfile for first deployment
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies for video processing and OpenCV runtime
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Copy local package source before installing requirements to support "-e ./revallusion_ai"
COPY revallusion_ai ./revallusion_ai

# Upgrade pip tooling for better resolution and wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Preinstall CPU-only PyTorch wheels from official index to avoid resolver conflicts
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchvision==0.18.1

# Copy constraints to pin heavy deps to compatible versions
COPY constraints.txt ./constraints.txt

# Install Python dependencies with constraints (includes editable local package)
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Run with uvicorn (simple, not gunicorn for first deployment)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]