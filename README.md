---
title: Biochem Spoilage Detect
emoji: 📊
colorFrom: pink
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: '3.13'
app_file: app.py


pinned: false
---

# Biochem Spoilage Detect

Detects medicine spoilage from images of syrup bottles and drug packaging using **MiniCPM-V 2.6 INT4** (8B params, bitsandbytes quantized) served via Modal. Upload a photo, crop to the label area to reduce visual tokens, and get a full spoilage analysis.

## What it does

1. **Extracts medicine info** — name, manufacturer, dates, ingredients, batch number
2. **Detects visual spoilage** — discoloration, cloudiness, sediment, seal damage (with suspension context rule)
3. **Estimates bacteria growth** — 0-100 scale based on visual cues and preservative analysis
4. **Lists chemical composition** — ingredients with risk levels
5. **Compares dates** — static expiry vs predicted spoilage date
6. **Visualizes everything** — chemical bar chart, timeline, bacteria gauge, risk radar

## Tech Stack

- **Model:** `openbmb/MiniCPM-V-2_6` (8B params, runtime INT4 quantized) served via vLLM on Modal
- **Frontend:** Gradio 6.17.3 on HuggingFace Spaces
- **Backend:** Modal serverless GPU (L4) with auto-scaling
- **Charts:** Plotly (interactive)
- **Image Cropping:** `gr.ImageEditor` for label-focused analysis (reduces visual tokens)

## Built for

[Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: Backyard AI

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
