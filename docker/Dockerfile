FROM python:3.7.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    gdown \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirement.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY . /app/

# Download TalkNet-ASD model from Google Drive
RUN cd /app/models && \
    gdown https://drive.google.com/uc?id=1jvjNNdfNLl-vTFX_0X-NHJ3IGBfTU44A -O TalkNet-ASD.zip && \
    unzip TalkNet-ASD.zip && \
    rm TalkNet-ASD.zip

# Create directory for shared data
RUN mkdir -p /shared_data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["tail", "-f", "/dev/null"]
