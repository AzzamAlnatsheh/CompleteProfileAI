---
title: CompleteProfile AI
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
---

# 💼 CompleteProfile AI (LinkedIn Optimizer AI)

CompleteProfile AI is an all-in-one digital career assistant designed to help job seekers build fully optimized LinkedIn profiles from scratch. For many new professionals, writing high-impact copy and securing polished, professional headshots are massive barriers to entry. 

This pure-Python, database-free application solves both problems by interviewing users to generate tailored, keyword-rich headlines and summaries, instantly stripping messy photo backgrounds for a studio finish, and generating custom-fit, abstract LinkedIn banners matching their industry.

---

## 🚀 Key Features

1. **AI Career Journalist (Text Optimizer):** A conversational chatbot interface powered by `gpt-4o-mini`. It interviews the user with dynamic prompts and translates raw, messy text into professional, keyword-optimized headlines and LinkedIn "About" summaries.
2. **Instant Studio Headshot Generator (Photo Editor):** An in-memory computer vision pipeline powered by the open-source **BiRefNet** model. It instantly detects and strips busy, casual backgrounds from user-uploaded portraits, allowing them to overlay a clean, solid color or professional studio backdrop.
3. **Context-Aware Banner Artist (Banner Generator):** A questionnaire-driven visual tool that prompts the **FLUX.1-schnell** image generation engine via the free Hugging Face Inference API. It outputs custom, beautifully scaled geometric abstract banners (**1584 x 396 px**) matched to the user’s industry and brand palette.

---

## 🛠️ Technical Stack

*   **Frontend UI:** [Gradio](https://www.gradio.app/) (utilizing a seamless, tabbed `gr.Blocks` layout).
*   **Hosting Platform:** [Hugging Face Spaces](https://huggingface.co/spaces) (Free CPU Basic Tier).
*   **Text Processing Engine:** OpenAI API (`gpt-4o-mini` with strict structured JSON output mapping).
*   **Background Remover Engine:** Local [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (loaded in-memory via PyTorch).
*   **Banner Engine:** Hugging Face Inference API calling `black-forest-labs/FLUX.1-schnell`.
*   **Image Processing:** Python Imaging Library (`Pillow` / PIL) for background-foreground compositing and canvas resizing.

---

## 💻 Local Setup & Installation

If you want to run this application locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://huggingface.co/spaces/Crackershoot/CompleteProfile-AI
cd CompleteProfile-AI
