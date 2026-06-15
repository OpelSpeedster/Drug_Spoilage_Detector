---
title: Drug Spoilage Detector
emoji: 📊
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: '3.13'
app_file: app.py
pinned: false
tags:
  - track:backyard
  - sponsor:openbmb
  - sponsor:modal
  - achievement:offgrid
  - achievement:offbrand
  - achievement:sharing
  - achievement:fieldnotes
---

# Drug Spoilage Detector

Detects medicine spoilage from images of syrup bottles and drug packaging using **MiniCPM-V 2.6 INT4** (8B params, bitsandbytes quantized) served via Modal. Upload a photo, crop to the label area to reduce visual tokens, and get a full spoilage analysis.

## What it does

1. **Extracts medicine info** — name, manufacturer, dates, ingredients, batch number
2. **Detects visual spoilage** — discoloration, cloudiness, sediment, seal damage (with suspension context rule)
3. **Estimates bacteria growth** — 0-100 scale based on visual cues and preservative analysis
4. **Lists chemical composition** — ingredients with quantities and risk levels
5. **Two-mode date comparison:**
   - **Mode 1 (dates found):** Static expiry as primary, visual estimate as secondary warning
   - **Mode 2 (no dates):** Visual spoilage score as primary estimate (for shopkeeper cut strips)
6. **Visualizes everything** — chemical bar chart, expiry timeline, bacteria gauge, growth curve, color degradation, risk radar

## Tech Stack

- **Model:** `openbmb/MiniCPM-V-2_6` (8B params, runtime INT4 quantized) served via vLLM on Modal
- **Frontend:** Gradio 6.17.3 on HuggingFace Spaces
- **Backend:** Modal serverless GPU (L4) with auto-scaling
- **Charts:** Plotly (interactive)
- **Image Preprocessing:** Contrast enhancement + sharpening for medicine labels

## VLM Pipeline

Two-pass architecture with parallel execution:

1. **Pass 1 — OCR:** Extract all visible text from medicine packaging
2. **Pass 2 — Analysis:** Four parallel VLM calls using OCR context:
   - Info extraction (name, dates, ingredients)
   - Spoilage assessment (visual indicators)
   - Bacteria risk estimate
   - Chemical composition

Python fallback calculations for bacteria growth curve, color degradation timeline, and dynamic expiry estimation.

## Links

[YouTube](https://youtu.be/fyKL3cSDxDo), [LinkedIn](https://www.linkedin.com/posts/vishal-s-v_drug-spoilage-detector-a-hugging-face-space-share-7472296588283912192-hWBY/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE7ZGVEB9uZmvexoy9SwcIfLeGOes3cz2Uw)

## Built for

[Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: Backyard AI
