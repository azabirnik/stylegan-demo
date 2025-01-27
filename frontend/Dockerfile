# Use the official Python image as the base
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the working directory
COPY . /app

# Expose port 8501 for the Streamlit application
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]

