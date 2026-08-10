"""
CompleteProfile AI (LinkedIn Optimizer AI)
--------------------------------------------------------------------------
An all-in-one digital career assistant that helps job seekers build fully 
optimized LinkedIn profiles from scratch using Gradio, OpenAI, and BiRefNet.

Author: Azzam Alnatsheh
License: MIT
"""

import os
import io
import json
import random
import logging
import requests
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

# Load local environment variables from a .env file (if it exists)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Force optimized CPU execution threading configurations to safeguard free-tier servers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from torchvision import transforms

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CompleteProfileAI")

# --- Smart Hardware Allocation (CPU / CUDA / MPS) ---
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Leverages GPU acceleration on local Apple Silicon Macs!

logger.info(f"Using execution hardware device: {DEVICE.upper()}")

# --- In-Memory ML Model Loading (BiRefNet) ---
BIREFNET_MODEL = None
try:
    from transformers import AutoModelForImageSegmentation
    logger.info("Initializing BiRefNet Segmentation Weights...")
    BIREFNET_MODEL = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        trust_remote_code=True
    )
    BIREFNET_MODEL.to(DEVICE)
    BIREFNET_MODEL.eval()
    logger.info("BiRefNet successfully loaded and compiled in-memory.")
except Exception as e:
    logger.error(f"Failed to initialize local BiRefNet: {e}. Activating transparent fallback.")

# --- HELPER 1: Programmatic Studio Gradient Generator ---
def create_gradient_backdrop(style="neutral_gray", size=(1024, 1024)):
    """Generates beautiful, professional linear gradient canvases directly in-memory."""
    width, height = size
    base = Image.new("RGB", size)
    
    if style == "office_blue":
        color1 = (15, 32, 67)      # Deep Corporate Navy
        color2 = (44, 83, 130)     # Clean Soft Slate Blue
    elif style == "soft_teal":
        color1 = (11, 40, 41)      # Dark Forest Teal
        color2 = (41, 108, 104)    # Warm Modern Muted Teal
    else: # neutral_gray
        color1 = (25, 25, 25)      # Rich Charcoal
        color2 = (85, 85, 85)      # Soft Studio Medium Gray
        
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        for x in range(width):
            base.putpixel((x, y), (r, g, b))
            
    return base

# --- HELPER 2: Programmatic Abstract Fallback Banners ---
def generate_fallback_banner(industry, color_palette_name):
    """Generates a beautiful geometric abstract banner programmatically if the API is offline."""
    size = (1584, 396)
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    
    # Established Corporate Brand Palettes
    palettes = {
        "Corporate Blue": ((15, 32, 67), (44, 83, 130), (100, 149, 237)),
        "Creative Teal": ((11, 40, 41), (41, 108, 104), (127, 255, 212)),
        "Tech Slate": ((15, 15, 15), (50, 50, 60), (150, 150, 160)),
        "Creative Amber": ((60, 30, 10), (120, 60, 20), (255, 191, 0)),
    }
    
    colors = palettes.get(color_palette_name, palettes["Corporate Blue"])
    c1, c2, c3 = colors
    
    # Render background gradient
    for y in range(396):
        ratio = y / 396
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        draw.line([(0, y), (1584, y)], fill=(r, g, b))
        
    # Generate abstract overlapping translucent shapes seeded by industry
    random.seed(hash(industry))
    for _ in range(12):
        x1 = random.randint(0, 1584)
        y1 = random.randint(0, 396)
        x2 = x1 + random.randint(100, 450)
        y2 = y1 + random.randint(50, 300)
        x3 = x1 + random.randint(-200, 200)
        y3 = y1 + random.randint(-150, 150)
        
        shape_img = Image.new("RGBA", size)
        shape_draw = ImageDraw.Draw(shape_img)
        fill_color = random.choice([c2, c3]) + (random.randint(25, 75),) # RGB + Alpha
        shape_draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=fill_color)
        image = Image.alpha_composite(image.convert("RGBA"), shape_img).convert("RGB")
        
    return image

# --- CORE FUNCTION 1: AI Career Journalist (Tab 1) ---
def optimize_linkedin_text(target_role, core_skills, achievement, style):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not configured.")
        return (
            "OpenAI API Key is missing. Please configure it in your .env or Environment Variables.",
            "Please configure your OPENAI_API_KEY setting inside your environment to activate the copywriter AI.",
            ["No Key Configuration Found"]
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system_prompt = """You are the "AI Career Journalist," an elite Executive Recruiter and world-class LinkedIn Copywriter. Your mission is to interview job seekers and translate their messy, unstructured, conversational raw inputs into compelling, high-converting, and keyword-optimized LinkedIn profiles. 

Your copywriting philosophy:
1. Cut the Fluff: Avoid generic corporate corporate-speak. Be concrete.
2. Quantify Impact: Turn passive duties into active, measurable achievements.
3. Keep it Human: Write in a natural, professional first-person tone ("I am...", "I lead...") that sounds like a confident professional, not an LLM.

You must strictly output your response in valid JSON format matching the schema requested. Do not write conversational preambles or postscripts."""

    task_prompt = f"""Transform the following conversational user inputs into a polished LinkedIn Headline, "About" Summary, and Keyword List.

### Few-Shot Example:
- USER INPUTS:
  * Target Role: Junior Software Engineer
  * Core Skills: Python, React, PostgreSQL, Git
  * Major Achievement: Built a campus tutoring app that was used by 300 students to schedule sessions.
  * Working Style: Collaborative, analytical, eager to solve complex logic.
- AI OUTPUT JSON:
{{
  "headline": "Junior Software Engineer | Python & React | Building Impact-Driven Web Solutions",
  "summary": "I am a software engineer focused on building highly functional, user-centric web applications. My passion lies in translating complex logic into clean, performant code.\\n\\nRecently, I developed a campus tutoring scheduler using Python and React, which successfully streamlined session booking for over 300 active student users. I thrive in collaborative environments where continuous learning and analytical problem-solving are valued.\\n\\nSpecialties: Python, JavaScript (React), SQL (PostgreSQL), Git, API Integration, and Agile Methodologies.",
  "extracted_keywords": ["Software Engineering", "Full-Stack Development", "Python", "React.js", "PostgreSQL", "Database Design", "Agile Methodologies"],
  "status": "success",
  "error_message": ""
}}

### Live Task:
- USER INPUTS:
  * Target Role: {target_role}
  * Core Skills: {core_skills}
  * Major Achievement: {achievement}
  * Working Style: {style}

Generate the JSON response following the exact schema shown in the examples. Your output MUST be pure JSON with no markdown wrapping (i.e. no ```json)."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" }, # Enforces structured JSON mapping
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_prompt}
            ],
            timeout=30
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        
        if data.get("status") == "error":
            return "", data.get("error_message", "Error compiling data."), []
            
        return data["headline"], data["summary"], data["extracted_keywords"]
    except Exception as e:
        logger.error(f"Error during OpenAI copy generation: {e}")
        return (
            f"{target_role} | {core_skills.split(',')[0] if core_skills else 'Professional'}",
            f"An error occurred while calling the OpenAI service: {e}. Please ensure your API secrets are configured correctly.",
            [s.strip() for s in core_skills.split(",")] if core_skills else []
        )

# --- CORE FUNCTION 2: Studio Headshot (Tab 2) ---
def process_headshot(image, bg_type, gradient_preset, solid_color):
    if image is None:
        return None
        
    # Safeguard against RAM overflow by enforcing a max dimension boundary
    max_size = 1024
    w, h = image.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        new_size = (int(w * ratio), int(h * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"Resized image boundary for safety: {new_size}")
        
    orig_w, orig_h = image.size
    
    # Segment Foreground Subject
    if BIREFNET_MODEL is None:
        logger.warning("BiRefNet uninitialized. Utilizing transparent bypass fallback.")
        cut_subject = image.convert("RGBA")
    else:
        try:
            logger.info("Initializing BiRefNet image segmentation...")
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_rgb = image.convert("RGB")
            input_tensor = transform_image(img_rgb).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = BIREFNET_MODEL(input_tensor)
                if hasattr(outputs, "logits"):
                    pred = outputs.logits[-1] if isinstance(outputs.logits, list) else outputs.logits
                elif isinstance(outputs, (list, tuple)):
                    pred = outputs[-1]
                else:
                    pred = outputs
                    
                if len(pred.shape) == 4:
                    pred = pred.squeeze(0).squeeze(0)
                elif len(pred.shape) == 3:
                    pred = pred.squeeze(0)
                    
                pred = torch.sigmoid(pred).cpu().numpy()
                
            mask = Image.fromarray((pred * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            cut_subject = image.convert("RGBA")
            cut_subject.putalpha(mask)
            logger.info("Headshot foreground segmentation completed successfully.")
        except Exception as e:
            logger.error(f"Error during segmentation pipeline: {e}")
            cut_subject = image.convert("RGBA")
            
    # Composite over Background Selection
    if bg_type == "Solid Color":
        hex_color = solid_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        background = Image.new("RGB", (orig_w, orig_h), rgb_color)
    else:
        preset_map = {
            "Neutral Charcoal Glow": "neutral_gray",
            "Corporate Navy Glow": "office_blue",
            "Modern Teal Glow": "soft_teal"
        }
        style = preset_map.get(gradient_preset, "neutral_gray")
        background = create_gradient_backdrop(style, size=(orig_w, orig_h))
        
    background.paste(cut_subject, (0, 0), cut_subject)
    return background

# --- CORE FUNCTION 3: Context-Aware Banner (Tab 3) ---
def process_banner(industry, color_palette):
    logger.info(f"Generating banner for Industry: '{industry}' with colors: '{color_palette}'")
    
    # Deploy to HF cloud inference API (FLUX.1-schnell model endpoints)
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    prompt = (
        f"A sleek, professional minimalist banner for the {industry} industry. "
        f"Abstract geometric art backdrop styling, color scheme: {color_palette}. "
        f"Clean lines, professional composition, modern graphic background, "
        f"no text, no letters, no logos, banner layout, 1584x396."
    )
    
    payload = {
        "inputs": prompt,
        "parameters": {"width": 1024, "height": 512}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            logger.info("Successfully fetched banner from FLUX API.")
            image = Image.open(io.BytesIO(response.content))
            return image.resize((1584, 396), Image.Resampling.LANCZOS)
        else:
            logger.warning(f"Flux API fallback activated. Response Status: {response.status_code}")
            return generate_fallback_banner(industry, color_palette)
    except Exception as e:
        logger.error(f"Flux generation failed: {e}. Executing fallback canvas generator.")
        return generate_fallback_banner(industry, color_palette)


# --- GRADIO LAYOUT ASSEMBLY ---
with gr.Blocks(
    title="CompleteProfile AI", 
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
) as demo:
    
    gr.Markdown(
        """
        # 💼 CompleteProfile AI
        ### Your All-in-One Digital Career & LinkedIn Optimizer
        Build a high-converting, recruiter-ready professional presence from scratch in under 5 minutes.
        """
    )
    
    # Tab 1: Text Copywriting Engine
    with gr.Tab("✍️ AI Career Journalist"):
        gr.Markdown(
            """
            ### 1. Tell us your career story
            Complete this quick interactive form. Our AI will translate your answers into keyword-rich, impact-driven summaries.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_role = gr.Textbox(
                    label="What is your Target Role?", 
                    placeholder="e.g., Junior Product Manager, Digital Marketing Specialist..."
                )
                input_skills = gr.Textbox(
                    label="Core Skills / Technologies (separated by commas)", 
                    placeholder="e.g., Python, SQL, Project Management, Agile..."
                )
                input_achievement = gr.Textbox(
                    label="What is one major professional or academic accomplishment?", 
                    placeholder="e.g., Managed an Instagram page growing traffic by 25%, built a campus portal...",
                    lines=3
                )
                input_st
