from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import asyncio

app = FastAPI()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing and postprocessing transforms
imsize = 512 if torch.cuda.is_available() else 256  # Use smaller size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()
])


def image_loader(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# Define the neural style transfer model
class StyleTransferModel(nn.Module):
    def __init__(self, style_img, content_img):
        super(StyleTransferModel, self).__init__()
        self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.content_img = content_img
        self.style_img = style_img
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.model, self.style_losses, self.content_losses = self.get_style_model_and_losses()

    def get_style_model_and_losses(self):
        cnn = self.cnn
        normalization_mean = self.normalization_mean
        normalization_std = self.normalization_std

        normalization = Normalization(normalization_mean, normalization_std).to(device)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # Increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                name = 'unknown_{}'.format(i)

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(self.content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # Trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def run_style_transfer(self, input_img, num_steps=300, style_weight=1000000, content_weight=1):
        optimizer = optim.LBFGS([input_img.requires_grad_()])

        run = [0]
        while run[0] <= num_steps:

            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                self.model(input_img)
                style_score = 0
                content_score = 0

                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                return loss

            optimizer.step(closure)

        # Clamp final output
        input_img.data.clamp_(0, 1)
        return input_img


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0).to(device)

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0).to(device)

    def gram_matrix(self, input):
        batch_size, feature_maps, h, w = input.size()
        features = input.view(batch_size * feature_maps, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch_size * feature_maps * h * w)

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Reshape mean and std to [C x 1 x 1]
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


async def perform_style_transfer(content_bytes, style_bytes, alpha):
    content_img = image_loader(content_bytes)
    style_img = image_loader(style_bytes)
    input_img = content_img.clone()

    model = StyleTransferModel(style_img, content_img)

    loop = asyncio.get_event_loop()
    stylized_output = await loop.run_in_executor(None, model.run_style_transfer, input_img)

    # Blend with original image based on alpha
    if alpha < 1.0:
        stylized_output = alpha * stylized_output + (1 - alpha) * content_img
        stylized_output.clamp_(0, 1)

    # Convert tensor to PIL Image
    unloader = transforms.ToPILImage()
    image = stylized_output.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    return image


@app.post("/style-transfer")
async def style_transfer(
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
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid content image")

    # Load style image
    try:
        style_bytes = await style_image.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid style image")

    # Perform style transfer asynchronously
    try:
        stylized_image = await perform_style_transfer(content_bytes, style_bytes, alpha)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

    # Save stylized image to bytes
    buf = io.BytesIO()
    stylized_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")