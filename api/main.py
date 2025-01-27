
m fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import os
import json
import asyncio

# Define paths
CYCLEGAN_DIR = "./CycleGAN"
MODEL_PATH = "./models/cyclegan/latest_net_G_A.pth"
OPTIONS_PATH = "./models/cyclegan/options.json"

# Add CycleGAN to the system path
import sys
sys.path.append(CYCLEGAN_DIR)

# Import CycleGAN utilities
from models import create_model
from options.test_options import TestOptions
from util import util

app = FastAPI()

# Initialize CycleGAN model at startup
@app.on_event("startup")
async def load_model():
    global model
    global device
    # Load options
    with open(OPTIONS_PATH, 'r') as f:
        opts = json.load(f)

    # Set up TestOptions
    test_opts = TestOptions().parse([])
    for key, value in opts.items():
        setattr(test_opts, key, value)

    # Create model
    model = create_model(test_opts)
    model.setup(test_opts)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.netG_A.to(device)

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize(int(256 * 1.12), transforms.InterpolationMode.BICUBIC),
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def perform_style_transfer_sync(content_image: Image.Image, alpha: float = 1.0) -> Image.Image:
    # Preprocess image
    input_tensor = transform(content_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        fake = model.netG_A(input_tensor)
        fake = (fake * 0.5 + 0.5).clamp(0, 1)
        fake = fake.cpu()

    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    stylized_image = to_pil(fake.squeeze(0))

    # Blend with original image based on alpha
    if alpha < 1.0:
        content_resized = content_image.resize(stylized_image.size)
        stylized_image = Image.blend(content_resized, stylized_image, alpha)

    return stylized_image

async def perform_style_transfer(content_image: Image.Image, alpha: float = 1.0) -> Image.Image:
    loop = asyncio.get_event_loop()
    stylized_image = await loop.run_in_executor(None, perform_style_transfer_sync, content_image, alpha)
    return stylized_image

@app.post("/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),  # Note: CycleGAN uses fixed style; style_image can be used to select model
    alpha: float = Form(...)
):
    # Validate alpha
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="Alpha must be between 0.0 and 1.0")

    # Load content image
    try:
        content_bytes = await content_image.read()
        content = Image.open(io.BytesIO(content_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid content image")

    # Load style image (optional: for selecting model based on style)
    # For simplicity, we'll ignore style_image as CycleGAN requires predefined models
    # Alternatively, implement a mapping from style_image to a specific CycleGAN model

    # Perform style transfer asynchronously
    stylized_image = await perform_style_transfer(content, alpha)

    # Save stylized image to bytes
    buf = io.BytesIO()
    stylized_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

