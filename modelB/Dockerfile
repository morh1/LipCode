# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential cmake libboost-all-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Environment fix for OpenCV on Windows
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Create processed and transcripts directories inside the container if not mounted
RUN mkdir -p /app/proccessed /app/transcripts

# Set default command to run your model script
CMD ["python", "modelB.py"]
