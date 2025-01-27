from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
import io
import sys
import os
import torchvision.transforms as transforms  # Added import statement

# Add AdaIN directory to the system path
sys.path.append('./pytorch-AdaIN')

from function import adaptive_instance_normalization
import net

app = FastAPI()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained models
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

# Load model weights
decoder.load_state_dict(torch.load('./pytorch-AdaIN/decoder.pth', map_location=device))
vgg.load_state_dict(torch.load('./pytorch-AdaIN/vgg_normalised.pth', map_location=device))

vgg = vgg.to(device)
decoder = decoder.to(device)

# Specify the layers to use
vgg = torch.nn.Sequential(*list(vgg.children())[:31])

# Image transformations
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

# Function to perform style transfer
def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert 0.0 <= alpha <= 1.0
    with torch.no_grad():
        content_features = vgg(content)
        style_features = vgg(style)
        t = adaptive_instance_normalization(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        output = decoder(t)
    return output

@app.post("/style-transfer")
async def style_transfer_endpoint(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    alpha: float = Form(1.0)
):
    # Validate alpha
    try:
        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="Alpha must be a float between 0.0 and 1.0")

    # Load content image
    try:
        content_bytes = await content_image.read()
        content = transform_image(content_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid content image")

    # Load style image
    try:
        style_bytes = await style_image.read()
        style = transform_image(style_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid style image")

    # Perform style transfer
    try:
        output = style_transfer(vgg, decoder, content, style, alpha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

    # Convert tensor to PIL Image
    output = output.cpu()
    output_image = transforms.ToPILImage()(output.squeeze(0))

    # Save stylized image to bytes
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")