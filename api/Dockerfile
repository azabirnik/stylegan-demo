# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone CycleGAN repository (optional: specify a commit or tag)
RUN git clone --depth 1 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git CycleGAN

# Copy main application files
COPY main.py .
COPY models ./models

# Ensure the models directory exists
RUN mkdir -p models/cyclegan

# Download pre-trained CycleGAN model
RUN wget -O models/cyclegan/latest_net_G_A.pth https://download.pytorch.org/models/cycle_gan/summer2winter/latest_net_G_A.pth

# Download options.yml
RUN wget -O models/cyclegan/options.yml https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/master/options/test/summer2winter.yml

# Convert options.yml to options.json
RUN python -c "import yaml, json; \
    with open('models/cyclegan/options.yml') as f_in: \
        data = yaml.safe_load(f_in); \
    with open('models/cyclegan/options.json', 'w') as f_out: \
        json.dump(data, f_out)"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]