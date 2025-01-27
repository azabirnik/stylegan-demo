# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git python3.12 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Clone AdaIN repository
RUN git clone https://github.com/naoto0804/pytorch-AdaIN.git
RUN mkdir models
RUN wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth -O models/decoder.pth
RUN wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth -O models/vgg_normalised.pth

# Copy main application files
COPY main.py .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
