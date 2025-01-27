# StyleGAN-Demo

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Backend Service](#backend-service)
    - [Building and Running the Backend](#building-and-running-the-backend)
    - [API Documentation](#api-documentation)
      - [Request Parameters](#request-parameters)
      - [Example Request](#example-request)
      - [Response](#response)
      - [OpenAPI Documentation](#openapi-documentation)
  - [Frontend Application](#frontend-application)
    - [Building and Running the Frontend](#building-and-running-the-frontend)
    - [Using the Frontend Application](#using-the-frontend-application)
    - [Using the Frontend with a Custom Backend](#using-the-frontend-with-a-custom-backend)
- [Configuration](#configuration)
  - [Backend Configuration](#backend-configuration)
  - [Frontend Configuration](#frontend-configuration)
- [Makefile Commands](#makefile-commands)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

**StyleGAN-Demo** is a project that deploys an image style transfer service using Docker for containerization. It consists of a backend API built with FastAPI and a frontend application built with Streamlit. Users can perform style transfer on images through an intuitive web interface or directly interact with the API for integration into other applications.

The style transfer process takes a **content image** and a **style image** as inputs and produces a new image that blends the content of the first image with the style of the second image using neural style transfer techniques.

## Features

- **Backend API**:
  - Performs style transfer using Neural Style Transfer with a pre-trained VGG19 model.
  - Utilizes GPU acceleration if available.
  - Processes both content and style images provided by the user.
  - Provides a RESTful API endpoint for easy integration.

- **Frontend Application**:
  - User-friendly interface built with Streamlit.
  - Allows users to upload content and style images.
  - Adjust the blending alpha parameter for style intensity.
  - Communicates with the backend API via HTTP requests.

- **Dockerized Deployment**:
  - Easy to set up and run with Docker and Docker Compose.
  - No need to install dependencies on the host machine.
  - Supports running frontend and backend on separate machines.

## Project Structure

```
stylegan-demo/
├── LICENSE
├── README.md
├── Makefile
├── api/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── main.py
│   ├── requirements.txt
└── frontend/
    ├── Dockerfile
    ├── docker-compose.yml
    ├── app.py
    └── requirements.txt
```

## Prerequisites

- **Docker**: Install Docker Engine on your machine. [Get Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Install Docker Compose. [Install Docker Compose](https://docs.docker.com/compose/install/)
- **NVIDIA Container Toolkit**: If you plan to use GPU acceleration, install the NVIDIA Container Toolkit. [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stylegan-demo.git
cd stylegan-demo
```

Ensure that you have the necessary permissions and tools installed to build and run Docker containers.

## Usage

You can run the backend and frontend services separately or together. Below are instructions for each.

### Backend Service

#### Building and Running the Backend

Navigate to the `api` directory:

```bash
cd api
```

Build and run the backend service using Docker Compose:

```bash
docker-compose up --build -d
```

**Note:** If using GPU acceleration, ensure that your Docker environment is configured to use GPUs.

The backend API will be accessible at `http://localhost:2026` (or the public IP if running on a remote server).

#### API Documentation

The backend API provides a `/style-transfer` endpoint for performing style transfer.

- **Endpoint**: `POST /style-transfer`

##### Request Parameters

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `content_image` (required): The content image file.
  - `style_image` (required): The style image file.
  - `alpha` (optional): A float between `0.0` and `1.0` indicating the blending factor between the content image and the stylized result. Default is `1.0`.

##### Example Request

```bash
curl -X POST http://localhost:2026/style-transfer \
     -F content_image=@path/to/your/content.jpg \
     -F style_image=@path/to/your/style.jpg \
     -F alpha=0.8 \
     --output stylized_output.png
```

##### Response

- **Success (200 OK)**: Returns the stylized image as a PNG file.
- **Error (4xx, 5xx)**: Returns an error message in JSON format.

##### OpenAPI Documentation

Access the interactive API documentation provided by FastAPI at:

```
http://localhost:2026/docs
```

### Frontend Application

#### Building and Running the Frontend

Navigate to the `frontend` directory:

```bash
cd frontend
```

Build and run the frontend service using Docker Compose:

```bash
docker-compose up --build -d
```

The frontend application will be accessible at `http://localhost:2025`.

#### Using the Frontend Application

1. Open your web browser and navigate to `http://localhost:2025`.
2. Upload a **content image**.
3. Upload a **style image**.
4. Enter the **REST API URL** for the backend service (e.g., `http://localhost:2026/style-transfer`).
5. Adjust the **alpha** parameter if desired.
6. Click on **Perform Style Transfer**.
7. The stylized image will be displayed, and you can download it using the provided button.

### Using the Frontend with a Custom Backend

If you have a backend service running elsewhere, you can use the frontend application to interact with it:

1. Ensure your backend service is accessible and the API URL is known.
2. In the frontend application, enter the backend API URL in the provided input field.
3. Proceed to perform style transfer as usual.

**Note:** Make sure that any network restrictions or firewalls allow communication between the frontend and backend services.

## Configuration

### Backend Configuration

The backend service configuration is managed via environment variables and Docker settings.

- **Port Mapping**: Adjust the port mapping in `api/docker-compose.yml` if necessary.

  ```yaml
  ports:
    - "2026:8000"  # Adjust host port as needed
  ```

- **GPU Support**: Ensure the `device_requests` section in `api/docker-compose.yml` is properly configured.

  ```yaml
  device_requests:
    - driver: nvidia
      count: 1
      capabilities: [gpu]
  ```

### Frontend Configuration

- **Default API URL**: The frontend allows users to input the backend API URL. You can set a default value in `app.py`.

  ```python
  api_url = st.sidebar.text_input(
      "REST API URL",
      value="http://localhost:2026/style-transfer",
      help="Enter the URL of the REST API endpoint for style transfer.",
  )
  ```

- **Port Mapping**: Adjust the port mapping in `frontend/docker-compose.yml` if necessary.

  ```yaml
  ports:
    - "2025:8501"  # Adjust host port as needed
  ```

## Makefile Commands

A `Makefile` is provided at the root of the project to simplify building and running the services.

- **Build and run the frontend service:**

  ```bash
  make frontend
  ```

- **Stop the frontend service:**

  ```bash
  make frontend-stop
  ```

- **Build and run the backend service:**

  ```bash
  make backend
  ```

- **Stop the backend service:**

  ```bash
  make backend-stop
  ```

- **Display help:**

  ```bash
  make help
  ```

**Note**: Ensure you are in the root directory of the project when running `make` commands.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Commit your changes with clear messages.
4. Push your branch to your fork.
5. Submit a pull request detailing your changes.

Before contributing, please ensure that your code adheres to the project's coding standards and passes any existing tests.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- **Neural Style Transfer**: This project uses neural style transfer techniques based on implementations from the [PyTorch Tutorials](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html).
- **FastAPI**: For providing a modern, fast (high-performance) web framework for building APIs with Python.
- **Streamlit**: For making it easy to build beautiful web apps for machine learning and data science.

---

**Disclaimer**: This project is for educational purposes. The models and code are provided as-is without warranty of any kind.