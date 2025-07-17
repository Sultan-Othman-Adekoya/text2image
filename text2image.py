
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

# ---- Cloudinary Configuration ----
cloudinary.config(
    cloud_name=st.secrets["CLOUDINARY"]["cloud_name"],
    api_key=st.secrets["CLOUDINARY"]["api_key"],
    api_secret=st.secrets["CLOUDINARY"]["api_secret"],
    secure=True
)

# ---- Cloudinary Upload Function ----
def upload_to_cloudinary(image_path):
    try:
        upload_result = cloudinary.uploader.upload(image_path)
        public_id = upload_result["public_id"]

        # Optional: Create optimized or transformed URLs
        optimized_url, _ = cloudinary_url(public_id, fetch_format="auto", quality="auto")
        auto_crop_url, _ = cloudinary_url(public_id, width=500, height=500, crop="auto", gravity="auto")

        return upload_result["secure_url"]  # or return optimized_url / auto_crop_url
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

# ---- Load Stable Diffusion Model ----
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_model()

# ---- Streamlit App UI ----
st.title("🎨 Text-to-Image Generator")
prompt = st.text_input("Enter your prompt (e.g. 'a cute cat on a bike')")

if st.button("Generate Image") and prompt:
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        image_path = "generated_image.png"
        image.save(image_path)

        # Upload image to Cloudinary
        image_url = upload_to_cloudinary(image_path)

    if image_url:
        st.image(image_url, caption="Generated via Stable Diffusion")
        st.markdown(f"[🔗 Click to view full image]({image_url})")
    else:
        st.error("❌ Failed to upload image.")
