# 💼 CompleteProfile AI (LinkedIn Optimizer AI)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gradio Version](https://img.shields.io/badge/Gradio-v5.0%2B-orange.svg)](https://www.gradio.app/)
[![OpenAI Model](https://img.shields.io/badge/OpenAI-gpt--4o--mini-green.svg)](https://openai.com/)
[![Deployment Status](https://img.shields.io/badge/Deployment-Free_Tiers-indigo.svg)](#-deployment-options)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CompleteProfile AI** is an all-in-one digital career assistant designed to help job seekers build fully optimized LinkedIn profiles from scratch. For new professionals or career switchers, writing high-impact copy and securing polished, professional headshots are massive, expensive barriers to entry. 

This pure-Python, database-free application solves both problems by interviewing users to generate tailored, keyword-rich headlines and summaries, instantly stripping messy photo backgrounds for a studio finish, and generating custom-fit, abstract LinkedIn banners matching their industry.

---

## 🚀 Key Features

*   **✍️ AI Career Journalist (Text Optimizer):** A conversational chatbot interface powered by `gpt-4o-mini`. It interviews the user with dynamic prompts and translates raw, messy text into professional, keyword-optimized headlines and LinkedIn "About" summaries using structured JSON output.
*   **📸 Instant Studio Headshot (Photo Editor):** An in-memory computer vision pipeline powered by the open-source **BiRefNet** model. It instantly detects and strips busy, casual backgrounds from user-uploaded portraits, allowing them to overlay a clean, solid color or professional studio gradient backdrop.
*   **🎨 Context-Aware Banner Artist (Banner Generator):** A questionnaire-driven visual tool that prompts the **FLUX.1-schnell** image generation engine via the free Hugging Face Inference API. It outputs custom, beautifully scaled geometric abstract banners (**1584 x 396 px**) matched to the user’s industry and brand palette, with a native Python procedural fallback engine if the API is offline.

---

## 🏗️ Minimum Working System (MWS) Architecture

The application is built around a stateless, in-memory processing pipeline to bypass database overhead and keep deployments incredibly lightweight and secure.

```text
  [ 1. INPUT LAYER ] ──► [ 2. PREPROCESSING ] ──► [ 3. CONTEXT/RETRIEVAL ]
   Gradio Frontend          In-Memory Setup         Local Python Dictionaries
  (Text/Photo/Industry)     (Resize & Sanitize)      (Few-Shot & Palette Maps)
                                                               │
 ┌─────────────────────────────────────────────────────────────┘
 │
 ▼
  [ 4. PROCESSING & PROMPTING ]
   ├── Text: GPT-4o-mini (OpenAI API Call)
   ├── Headshot: BiRefNet (Local PyTorch Segmenter on CPU)
   └── Banner: FLUX.1-schnell (HF Inference API Call / Fallback)
                               │
                               ▼
  [ 5. OUTPUT FORMATTING ] ─────────► [ 6. LIGHTWEIGHT LOGGING ]
   In-Memory PIL Assembly              Console stdout 
   Gradio Copy & Download Components   (No Persistent User File Storage)
