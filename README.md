# StyleGAN Demo

## Overview

**StyleGAN Demo** is a web application that enables users to perform style transfer on images using a GAN-based model. The application consists of a **Streamlit frontend** for uploading images and adjusting parameters, and a **FastAPI backend** that handles the style transfer inference.

## Features

- **Image Upload**: Upload a content image and a style image in JPG or PNG formats.
- **Parameter Adjustment**: Adjust the `alpha` parameter to control the influence of the style on the content.
- **Asynchronous Processing**: Utilizes `aiohttp` for non-blocking API requests, ensuring a responsive user experience.
- **Downloadable Results**: View and download the stylized image directly from the application.

## Project Structure

```
stylegan-demo/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app.py
├── README.md
└── LICENSE
```

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) installed.
- Access to the REST API endpoint for style transfer inference.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/stylegan-demo.git
   cd stylegan-demo
   ```

2. **Configure the API Endpoint**

   - Open `app.py` and navigate to the sidebar section.
   - Enter the URL of your REST API endpoint in the **API Configuration** section.

3. **Build and Run with Docker Compose**

   Ensure that your `docker-compose.yml` and `Dockerfile` are correctly set up as per the project structure.

   ```bash
   sudo docker-compose up --build
   ```

   This command will build the Docker image and start the Streamlit application on port `2025`.

4. **Access the Application**

   Open your web browser and navigate to:

   ```
   http://localhost:2025
   ```

   Replace `localhost` with your server's IP address if running on a remote machine.

## Usage

1. **Upload Images**

   - **Content Image**: Click on "Choose a Content Image" to upload the base image you want to stylize.
   - **Style Image**: Click on "Choose a Style Image" to upload the style you want to apply to the content image.

2. **Adjust Parameters**

   - Use the **Alpha Slider** in the sidebar to set the influence level of the style image on the content image (ranging from `0.0` to `1.0`).

3. **Perform Style Transfer**

   - Click on the **"Perform Style Transfer"** button.
   - Wait for the processing to complete. A spinner will indicate the ongoing process.
   - Once completed, the stylized image will be displayed along with an option to download it.

4. **Download Output**

   - Click the **"Download Output Image"** button to save the stylized image to your device.

## API Team Requirements

For the team handling the model and API for inference, please refer to the [API Team Requirements](#api-team-requirements-for-style-transfer-inference) section below.

## API Team Requirements for Style Transfer Inference

### 1. `requirements.txt`

```plaintext
fastapi==0.95.2
uvicorn==0.22.0
Pillow==9.5.0
aiofiles==23.1.0
torch==2.0.1
torchvision==0.15.2
```

### 2. API Endpoint Specification

- **Endpoint URL:** `/style-transfer`
- **Method:** `POST`
- **Request Payload:**
  - `content_image`: Image file (multipart/form-data)
  - `style_image`: Image file (multipart/form-data)
  - `alpha`: Float (0.0 to 1.0) (multipart/form-data)
- **Response:**
  - Stylized image file (`image/png`)

### 3. Example `main.py` Structure

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
# Import your style transfer model here

app = FastAPI()

@app.post("/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    alpha: float = Form(...)
):
    # Load and preprocess images
    content = Image.open(io.BytesIO(await content_image.read())).convert("RGB")
    style = Image.open(io.BytesIO(await style_image.read())).convert("RGB")
    
    # Perform style transfer (replace with your model's inference)
    stylized_image = perform_style_transfer(content, style, alpha)
    
    # Save stylized image to bytes
    buf = io.BytesIO()
    stylized_image.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")
```

### 4. Docker Setup (Optional)

#### `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Running the API

- **Without Docker:**

  ```bash
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

- **With Docker:**

  ```bash
  docker build -t style-transfer-api .
  docker run -d -p 8000:8000 style-transfer-api
  ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or support, please contact [your.email@example.com](mailto:your.email@example.com).
