import os
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(
    page_title="Style Transfer with GANs",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title and Description
st.title("Style Transfer Application")
st.write(
    """
    Upload a **content image** and a **style image**, adjust the **alpha parameter**, and perform style transfer using a remote GAN-based model.
    """
)

# Sidebar for Parameters and API Configuration
st.sidebar.header("Parameters")
alpha = st.sidebar.slider(
    "Alpha",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.01,
    help="Adjust the influence of the style image on the content image.",
)

st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input(
    "REST API URL",
    value=os.environ.get("API_URL", ""),
    help="Enter the URL of the REST API endpoint for style transfer.",
)


# Function to load and display images
def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


# Function to resize images
def resize_image(image, max_size=512):
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size)
    return image


# Function to send request to REST API
def send_request(content_image, style_image, alpha, api_url):
    """
    Sends the content image, style image, and alpha to the REST API synchronously using requests.

    Parameters:
        content_image (PIL.Image): The content image.
        style_image (PIL.Image): The style image.
        alpha (float): The alpha parameter.
        api_url (str): The REST API endpoint URL.

    Returns:
        PIL.Image or None: The stylized image if successful, None otherwise.
    """
    # Convert images to bytes
    buf_content = BytesIO()
    content_image.save(buf_content, format="PNG")
    content_bytes = buf_content.getvalue()

    buf_style = BytesIO()
    style_image.save(buf_style, format="PNG")
    style_bytes = buf_style.getvalue()

    # Prepare multipart/form-data payload
    files = {
        'content_image': ('content.png', content_bytes, 'image/png'),
        'style_image': ('style.png', style_bytes, 'image/png'),
    }
    data = {
        'alpha': str(alpha),
    }

    try:
        response = requests.post(api_url, files=files, data=data, timeout=120)
        if response.status_code == 200:
            output_image = Image.open(BytesIO(response.content)).convert("RGB")
            return output_image
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request to the API timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while requesting the API: {e}")
        return None


# Upload Content Image
st.header("1. Upload Content Image")
content_image_file = st.file_uploader(
    "Choose a Content Image", type=["jpg", "jpeg", "png"], key="content_uploader"
)
if content_image_file is not None:
    content_image = load_image(content_image_file)
    if content_image:
        content_image = resize_image(content_image)
        st.image(content_image, caption="Content Image", use_container_width=True)
else:
    content_image = None

# Upload Style Image
st.header("2. Upload Style Image")
style_image_file = st.file_uploader(
    "Choose a Style Image", type=["jpg", "jpeg", "png"], key="style_uploader"
)
if style_image_file is not None:
    style_image = load_image(style_image_file)
    if style_image:
        style_image = resize_image(style_image)
        st.image(style_image, caption="Style Image", use_container_width=True)
else:
    style_image = None

# Perform Style Transfer Button
if st.button("Perform Style Transfer"):
    if content_image is None or style_image is None:
        st.error("Please upload both content and style images.")
    elif not api_url:
        st.error("Please enter the REST API URL.")
    else:
        try:
            with st.spinner("Performing style transfer..."):
                output_image = send_request(content_image, style_image, alpha, api_url)
            if output_image:
                st.success("Style transfer completed!")
                st.image(output_image, caption="Output Image", use_container_width=True)
                # [Download button code]
            else:
                st.error("Failed to get a valid response from the API.")
        except Exception as e:
            st.error(f"An error occurred: {e}")