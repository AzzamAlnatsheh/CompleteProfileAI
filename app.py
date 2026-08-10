"""
CompleteProfile AI - Streamlit Edition
--------------------------------------------------------------------------
An all-in-one digital career assistant that helps job seekers build fully 
optimized LinkedIn profiles from scratch using Streamlit, OpenAI, and BiRefNet.
"""

import os
import io
import json
import random
import logging
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Ensure PyTorch CPU constraints for limited-RAM environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from torchvision import transforms

# --- Initialize Session State for AI Text Persistence ---
if "headline" not in st.session_state:
    st.session_state.headline = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")
logger = logging.getLogger("CompleteProfileAI")

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

# --- Cache Model in Streamlit (Crucial for 512MB RAM bounds!) ---
@st.cache_resource
def load_birefnet():
    try:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", 
            trust_remote_code=True
        )
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading BiRefNet: {e}")
        return None

BIREFNET_MODEL = load_birefnet()

# --- Helper Functions ---
def create_gradient_backdrop(style="neutral_gray", size=(1024, 1024)):
    width, height = size
    base = Image.new("RGB", size)
    if style == "office_blue":
        color1, color2 = (15, 32, 67), (44, 83, 130)
    elif style == "soft_teal":
        color1, color2 = (11, 40, 41), (41, 108, 104)
    else:
        color1, color2 = (25, 25, 25), (85, 85, 85)
        
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        for x in range(width):
            base.putpixel((x, y), (r, g, b))
    return base

def generate_fallback_banner(industry, color_palette_name):
    size = (1584, 396)
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    palettes = {
        "Corporate Blue": ((15, 32, 67), (44, 83, 130), (100, 149, 237)),
        "Creative Teal": ((11, 40, 41), (41, 108, 104), (127, 255, 212)),
        "Tech Slate": ((15, 15, 15), (50, 50, 60), (150, 150, 160)),
        "Creative Amber": ((60, 30, 10), (120, 60, 20), (255, 191, 0)),
    }
    colors = palettes.get(color_palette_name, palettes["Corporate Blue"])
    c1, c2, c3 = colors
    for y in range(396):
        ratio = y / 396
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        draw.line([(0, y), (1584, y)], fill=(r, g, b))
        
    random.seed(hash(industry))
    for _ in range(12):
        x1, y1 = random.randint(0, 1584), random.randint(0, 396)
        x2, y2 = x1 + random.randint(100, 450), y1 + random.randint(50, 300)
        x3, y3 = x1 + random.randint(-200, 200), y1 + random.randint(-150, 150)
        shape_img = Image.new("RGBA", size)
        shape_draw = ImageDraw.Draw(shape_img)
        fill_color = random.choice([c2, c3]) + (random.randint(25, 75),)
        shape_draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=fill_color)
        image = Image.alpha_composite(image.convert("RGBA"), shape_img).convert("RGB")
    return image

# --- OpenAI Copywriting Logic ---
def optimize_linkedin_text(target_role, core_skills, achievement, style):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API Key is missing.", "Configure your key in Streamlit App secrets.", []
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    system_prompt = "You are an elite LinkedIn copywriter. Output raw JSON containing fields 'headline', 'summary', and 'extracted_keywords' only."
    task_prompt = f"Target Role: {target_role}, Skills: {core_skills}, Achievement: {achievement}, Style: {style}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": task_prompt}]
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("headline", ""), data.get("summary", ""), data.get("extracted_keywords", [])
    except Exception as e:
        return f"{target_role} | Specialized", f"Error connecting to service: {e}", []

# --- Segmentation logic ---
def process_headshot(image, bg_type, gradient_preset, solid_color):
    if image is None: return None
    max_size = 1024
    w, h = image.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    orig_w, orig_h = image.size
    if BIREFNET_MODEL is None:
        cut_subject = image.convert("RGBA")
    else:
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform_image(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = BIREFNET_MODEL(input_tensor)
            pred = torch.sigmoid(outputs).cpu().numpy().squeeze()
        mask = Image.fromarray((pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        cut_subject = image.convert("RGBA")
        cut_subject.putalpha(mask)
    if bg_type == "Solid Color":
        hex_color = solid_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        background = Image.new("RGB", (orig_w, orig_h), rgb_color)
    else:
        preset_map = {"Neutral Charcoal Glow": "neutral_gray", "Corporate Navy Glow": "office_blue", "Modern Teal Glow": "soft_teal"}
        background = create_gradient_backdrop(preset_map.get(gradient_preset, "neutral_gray"), size=(orig_w, orig_h))
    background.paste(cut_subject, (0, 0), cut_subject)
    return background

# --- STREAMLIT UI DESIGN ---
st.set_page_config(page_title="CompleteProfile AI", layout="wide")
st.title("💼 CompleteProfile AI")
st.write("### Your All-in-One Digital Career & LinkedIn Optimizer")

tab1, tab2, tab3 = st.tabs(["✍️ AI Career Journalist", "🖼️ Instant Studio Headshot", "🎨 Context-Aware Banner"])

with tab1:
    st.write("### 1. Tell us your career story")
    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("What is your Target Role?", placeholder="e.g., Junior Product Manager")
        skills = st.text_input("Core Skills (separated by commas)")
        achievement = st.text_area("What is one major professional/academic accomplishment?")
        style = st.selectbox("Working Style Preset", ["Professional & Direct", "Creative & Innovative", "Warm & Collaborative"])
        btn = st.button("Generate Professional Copy", type="primary")
    with col2:
        if btn:
            with st.spinner("Writing optimized content..."):
                h_out, s_out, k_out = optimize_linkedin_text(role, skills, achievement, style)
                st.session_state.headline = h_out
                st.session_state.summary = s_out
                st.session_state.keywords = k_out
        
        # Elements pull from session state so data does not vanish when changing parameters
        st.text_input("Optimized LinkedIn Headline", value=st.session_state.headline)
        st.text_area("LinkedIn 'About' Summary", value=st.session_state.summary, height=250)
        st.multiselect("Extracted SEO Keywords", options=st.session_state.keywords, default=st.session_state.keywords)

with tab2:
    st.write("### 2. Instant Studio Backdrop Editor")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Casual Headshot", type=["jpg", "png", "jpeg"])
        bg_type = st.radio("Backdrop Type", ["Solid Color", "Gradient Studio Backdrop"])
        
        if bg_type == "Solid Color":
            solid_color = st.color_picker("Pick a background color", "#1E3A8A")
            gradient_style = None
        else:
            gradient_style = st.selectbox("Select Gradient Preset", ["Neutral Charcoal Glow", "Corporate Navy Glow", "Modern Teal Glow"])
            solid_color = None
            
        process_btn = st.button("Process Studio Portrait", type="primary")
    with col2:
        if uploaded_file and process_btn:
            with st.spinner("Extracting background with AI..."):
                input_img = Image.open(uploaded_file)
                output_img = process_headshot(input_img, bg_type, gradient_style, solid_color)
                
                if output_img:
                    st.image(output_img, caption="Your New Studio Profile Picture", width=400)
                    
                    # Convert file to buffer for download handling
                    buf = io.BytesIO()
                    output_img.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Profile Image",
                        data=byte_im,
                        file_name="linkedin_headshot.jpg",
                        mime="image/jpeg"
                    )

with tab3:
    st.write("### 3. Abstract Banner Artist")
    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox(
            "Your Target Industry", 
            ["Tech & Software", "Finance & Corporate", "Creative Design", "Healthcare", "Education & Non-Profit"]
        )
        palette = st.radio(
            "Brand Palette Style", 
            ["Corporate Blue", "Creative Teal", "Tech Slate", "Creative Amber"]
        )
        btn_banner = st.button("Generate Custom Banner", type="primary")
        
    with col2:
        if btn_banner:
            with st.spinner("Rendering your professional abstract banner..."):
                # Configure API requests to Hugging Face Inference cloud for Flux
                api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
                headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
                
                prompt = (
                    f"A sleek, professional minimalist banner for the {industry} industry. "
                    f"Abstract geometric art backdrop styling, color scheme: {palette}. "
                    f"Clean lines, professional composition, modern graphic background, "
                    f"no text, no letters, no logos, banner layout, 1584x396."
                )
                
                payload = {
                    "inputs": prompt,
                    "parameters": {"width": 1024, "height": 512}
                }
                
                banner_img = None
                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        banner_img = Image.open(io.BytesIO(response.content))
                        banner_img = banner_img.resize((1584, 396), Image.Resampling.LANCZOS)
                        logger.info("Successfully rendered custom banner with FLUX Cloud API.")
                    else:
                        logger.warning(f"FLUX cloud returned non-200 code: {response.status_code}. Initiating procedural generator.")
                except Exception as e:
                    logger.error(f"Failed to communicate with FLUX API: {e}. Running fallback generator.")
                
                # Check if fallback is required
                if banner_img is None:
                    banner_img = generate_fallback_banner(industry, palette)
                
                # Display output and enable seamless download
                st.image(banner_img, caption="Custom Banner (Scaled to LinkedIn standard 1584 x 396 px)", use_container_width=True)
                
                # Format to buffer
                buf_banner = io.BytesIO()
                banner_img.save(buf_banner, format="PNG")
                byte_banner = buf_banner.getvalue()
                
                st.download_button(
                    label="📥 Download Banner Image",
                    data=byte_banner,
                    file_name="linkedin_custom_banner.png",
                    mime="image/png"
                )
