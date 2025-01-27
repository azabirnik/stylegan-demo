# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Clone CycleGAN repository
RUN git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git CycleGAN

# Copy main application files
COPY main.py .
COPY models ./models

# (Optional) Download pre-trained CycleGAN model
# Replace the URL with the actual URL of the pre-trained model you intend to use
RUN mkdir -p models/cyclegan
# Example: Downloading the summer2winter model
RUN wget -O models/cyclegan/latest_net_G_A.pth https://download.pytorch.org/models/cycle_gan/summer2winter/latest_net_G_A.pth
# Download and convert options.json if necessary
RUN wget https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/master/options/test/summer2winter.yml -O models/cyclegan/options.yml && \
    apt-get update && apt-get install -y python3-pip && \
    pip install pyyaml && \
    python3 -c "import yaml, json, sys; \
               with open('models/cyclegan/options.yml') as f: \
                   data = yaml.safe_load(f); \
               with open('models/cyclegan/options.json', 'w') as f: \
                   json.dump(data, f)"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

