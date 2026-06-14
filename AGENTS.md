# AGENTS.md — Biochem Spoilage Detect

## Project Context

Hackathon entry for [Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: **Backyard AI**.
Detects medicine spoilage from images of syrup bottles and drug packaging using a small VLM.
Submission deadline: **June 15, 2026**.

## Tech Stack

- **Model:** `openbmb/MiniCPM-V-2_6` (8B params, runtime bitsandbytes INT4 quantized, served via vLLM on Modal)
- **Frontend:** Gradio 6.17.3 on HuggingFace Spaces
- **Backend:** Modal serverless GPU (L4) with vLLM inference
- **Python:** 3.13
- **Charts:** Plotly (interactive, renders in `gr.Plot`)
- **Entry point:** `app.py` (declared in README.md frontmatter)

## Architecture

```
HF Spaces (Gradio)                 Modal (GPU)
┌──────────────────┐               ┌─────────────────────────┐
│  app.py          │  HTTPS POST   │  backend.py             │
│  engine.py  ─────┼──────────────►│  vLLM + MiniCPM-V 2.6   │
│  prompts.py      │  JSON resp    │  INT4, L4 GPU, scales 0 │
│  utils.py        │◄──────────────│  @fastapi_endpoint      │
│  visualization.py│               │                         │
└──────────────────┘               └─────────────────────────┘
```

- **Frontend** (HF Spaces): Gradio UI, chart generation, spoilage scoring — no GPU needed
- **Backend** (Modal): VLM inference only — spins up on request, scales to 0 when idle
- **Communication:** Base64-encoded JPEG images sent via HTTPS to Modal endpoint (single image, cropped to label via ImageEditor)

## Hackathon Constraints

| Rule | Value |
|------|-------|
| Max model params | ≤ 32B (we use 8B, INT4 quantized) |
| App type | Gradio on HuggingFace Spaces |
| Submission | Space link + demo video + social post |
| No cloud APIs | Inference runs on Modal GPU (not a cloud API service like OpenAI) |

### Merit Badges Targeted

- **Off the Grid** — inference runs on Modal GPU (no third-party API service)
- **Sharing is Caring** — publish agent trace to HF Hub
- **Off-Brand** — custom Gradio theme via `gr.themes`

## File Architecture

```
app.py              # Gradio Blocks UI — main entry point
backend.py          # Modal deployment: vLLM + MiniCPM-V 2.6 INT4 on L4 GPU
engine.py           # HTTPS client that calls Modal endpoint (two-pass pipeline)
prompts.py          # VLM prompt templates with OCR context injection
utils.py            # Date parsing, chemical reference DB, spoilage scoring
visualization.py    # Plotly charts: chemical bar, timeline, gauge, radar
requirements.txt    # Lightweight: requests, gradio, plotly, pandas, etc.
README.md           # HF Spaces frontmatter (already configured)
agent_trace/        # Published trace for Sharing is Caring badge
```

## VLM Analysis Pipeline

Two-pass architecture for better text extraction from medicine packaging:

**Pass 1 — OCR Extraction:**
1. **PROMPT_OCR** — Read ALL visible text from every surface (front label, side panel, back, stickers, cap, foil). Returns structured JSON with text blocks, packaging type, and visible surfaces. Focuses on dot-matrix/stamped text for batch numbers and dates.

**Pass 2 — Structured Analysis** (uses OCR text as context):
1. **PROMPT_INFO** — Extract medicine info (name, manufacturer, dates, ingredients, storage). Includes date-guessing warning to set null when dates aren't visible.
2. **PROMPT_SPOILAGE** — Assess visual spoilage with pharmacological context rule to avoid false positives on suspensions.
3. **PROMPT_BACTERIA** — Bacteria growth estimate with preservative analysis from text. Ignores natural suspension opacity.
4. **PROMPT_CHEMICALS** — List chemicals with quantity, category, and risk level

All prompts instruct the model to return **valid JSON only**. Regex fallback parser handles non-JSON outputs. The OCR text is injected into Pass 2 prompts via `{OCR_TEXT}` placeholder.

**Optimization:** Pass 2 runs all 4 prompts in parallel via `ThreadPoolExecutor` (~4x speedup).

**Prompt format:** Prompts use `{OCR_TEXT}` placeholder which is filled via Python's `.format()`. JSON examples in prompts must use escaped braces `{{ }}` to avoid `KeyError`.

## Two-Mode Date System

The app operates in two modes based on whether dates are visible on the medicine label:

| | Mode 1 (dates found) | Mode 2 (no dates) |
|---|---|---|
| **Primary verdict** | Static expiry date | Visual spoilage score |
| **Timeline shows** | MFG → Today → Expiry | Today → Visual estimate |
| **Predicted spoilage** | Secondary note only | Primary estimate |
| **Verdict wording** | "X days until expiry" | "Visually OK / Spoiled" |
| **Use case** | Normal purchase | Shopkeeper cut strip |

**Mode detection:** If `exp_date` is found → Mode 1. If `exp_date` is null → Mode 2.

## Spoilage Scoring

```
score = (visual_indicators * 0.35) + (bacteria_level * 0.25) + (date_proximity * 0.20) + (color_deviation * 0.10) + (dynamic_expiry * 0.10)
```

Thresholds: >60 = SPOILED, 30-60 = WARNING, <30 = SAFE

## Date Parsing

`parse_date()` in `utils.py` handles multiple formats:
- DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY
- MM/YYYY, MM-YYYY, YYYY only
- Month YYYY (e.g., "SEP 2025"), DD Month YYYY
- **Dot-normalized:** "SEP.2025" → "SEP 2025" (dots replaced with spaces before parsing)
- **Pattern priority:** Month YYYY comes before YYYY-only to prevent incorrect matching

## Python Fallback Calculations

When VLM can't extract certain data, Python calculates:
1. **Bacteria growth curve** — Logistic model based on ingredients, preservatives, spoilage level
2. **Color degradation timeline** — Exponential decay based on visual indicators
3. **Dynamic expiry** — Adjusted shelf life based on visual damage, preservatives, color deviation

## Key Gotchas

- Model repo is `openbmb/MiniCPM-V-2_6-int4` (note the underscore in `2_6`).
- vLLM requires `--quantization bitsandbytes --load-format bitsandbytes` for INT4 models.
- Modal endpoint URL is set via `MODAL_ENDPOINT_URL` environment variable (HF Spaces secret).
- Base64 encoding increases image payload ~33% — JPEG quality 95 preserves fine print.
- Modal cold start takes ~30-60 sec after idle. First request may be slow.
- vLLM serves an OpenAI-compatible API — no custom client code needed.
- Plotly charts render in `gr.Plot()` components — not `gr.Image()`.
- The Modal backend (`backend.py`) is deployed separately from the HF Space.
- Image preprocessing enhances contrast + sharpens for medicine labels.
- Using `uv` instead of `pip` for faster Modal image builds.
- OCR text truncated to 800 chars in Pass 2 prompts (up from 300) to capture more date/ingredient info.
- OCR exceptions now propagate to error handler instead of failing silently.
- Dynamic expiry defaults to 1-year shelf life when dates are missing.
- `format_ocr_for_prompt` max_chars=800 for better context.
- Pass 2 prompts use `ThreadPoolExecutor` for parallel execution.
- User text (from "Additional Info" box) takes priority over OCR text in prompts.
- Chemical chart shows Name → Quantity with bar width proportional to parsed values.
- `.env` file contains `TOKEN` (HF token) and `MODAL_ENDPOINT_URL` (Modal endpoint).
- Modal secret created: `modal secret create huggingface TOKEN=hf_xxx`.
- Modal deployment requires `$env:PYTHONIOENCODING="utf-8"` on Windows.

## Running Locally

```bash
# Frontend only (needs MODAL_ENDPOINT_URL set)
pip install -r requirements.txt
MODAL_ENDPOINT_URL=https://your-workspace--biochem-spoilage-detect-analyze.modal.run python app.py
# Opens at http://localhost:7860
```

## Deploying

### 1. Deploy Modal backend

```bash
pip install modal
$env:PYTHONIOENCODING="utf-8"  # Windows only
modal deploy backend.py
# Prints the endpoint URL — save it
```

### 2. Deploy HF Space frontend

1. Set `MODAL_ENDPOINT_URL` as a Space secret (Settings → Secrets)
2. Push code to the HF Space
3. README.md frontmatter already declares `sdk: gradio`, `app_file: app.py`

## Submission Checklist

- [x] Modal backend deployed and endpoint URL saved
- [x] HF Space deployed with MODAL_ENDPOINT_URL secret configured
- [ ] Demo video recorded (show: upload image → crop label → analyze → charts + verdict)
- [ ] Social media post made
- [ ] Agent trace published to HF Hub (Sharing is Caring badge)
