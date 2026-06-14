"""VLM analysis pipeline via Modal inference backend.

Architecture:
  Pass 1 — OCR extraction: Read visible text from all images.
  Pass 2 — Structured analysis: Use OCR text for 4 short prompts.
  Fallback — Python calculations for bacteria growth, color, dynamic expiry.

Retry mechanism: Up to 7 retries with progressive prompt simplification.
"""

import base64
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests
from PIL import Image

from prompts import PROMPT_OCR, PASS2_PROMPTS, format_ocr_for_prompt
from utils import (
    parse_date,
    calculate_theoretical_growth,
    calculate_dynamic_expiry,
    estimate_color_from_spoilage,
)

MODAL_ENDPOINT_URL = os.environ.get(
    "MODAL_ENDPOINT_URL",
    "https://vishalsv2205--biochem-spoilage-detect-vlminference-analyze.modal.run",
)

MAX_RETRIES = 3


@dataclass
class AnalysisResult:
    """Structured result from VLM analysis."""

    ocr_data: dict = field(default_factory=dict)
    medicine_info: dict = field(default_factory=dict)
    spoilage_assessment: dict = field(default_factory=dict)
    bacteria_estimate: dict = field(default_factory=dict)
    chemicals: list = field(default_factory=list)
    raw_responses: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    bacteria_growth_curve: dict = field(default_factory=dict)
    color_analysis: dict = field(default_factory=dict)
    dynamic_expiry: dict = field(default_factory=dict)


def _image_to_base64(image) -> str:
    """Convert image to base64-encoded JPEG string.
    
    Optimized for MiniCPM-V 2.6:
    - Resizes to max 1344x1344 (optimal for 640 token density)
    - Ensures RGB mode for JPEG compatibility
    - Uses quality 95 for fine print readability
    """
    if isinstance(image, tuple):
        image = image[0]
    if not isinstance(image, Image.Image):
        try:
            import numpy as np
            image = Image.fromarray(np.array(image))
        except Exception as e:
            raise ValueError(f"Cannot convert image to PIL: {e}")
    
    # Convert to RGB if needed
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    # Optimize resolution for MiniCPM-V 2.6 (max 1344x1344 for 640 tokens)
    max_size = 1344
    if image.width > max_size or image.height > max_size:
        # Maintain aspect ratio
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from VLM response with multiple fallback strategies."""
    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown code blocks
    for pattern in [r"```json\s*\n?(.*?)\n?\s*```", r"```\s*\n?(.*?)\n?\s*```"]:
        fence_match = re.search(pattern, cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
            break

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    return None


def _preprocess_images_for_medicine(images: list) -> list:
    """Preprocess images specifically for medicine label analysis.
    
    Optimizations for medicine labels:
    - Resizes to max 1344x1344 (optimal for MiniCPM-V 2.6)
    - Enhances contrast for better OCR
    - Sharpens text for clearer reading
    - Optimizes for small print and dot-matrix text
    """
    processed = []
    for img in images:
        if isinstance(img, tuple):
            img = img[0]
        if not isinstance(img, Image.Image):
            try:
                import numpy as np
                img = Image.fromarray(np.array(img))
            except:
                processed.append(img)
                continue
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Optimize resolution for MiniCPM-V 2.6 (max 1344x1344)
        max_size = 1344
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply enhancements for better OCR
        from PIL import ImageEnhance, ImageFilter
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Sharpen for better text reading
        img = img.filter(ImageFilter.SHARPEN)
        
        processed.append(img)
    
    return processed


def _run_prompt(images: list, prompt: str) -> str:
    """Run a single prompt on images via the Modal endpoint."""
    if not MODAL_ENDPOINT_URL:
        raise RuntimeError("MODAL_ENDPOINT_URL not set. Deploy backend first.")

    b64_list = [_image_to_base64(img) for img in images[:4]]
    resp = requests.post(
        MODAL_ENDPOINT_URL,
        params={"prompt": prompt},
        json=b64_list,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _simplify_prompt(prompt: str, retry_num: int) -> str:
    """Progressively simplify prompt on each retry."""
    if retry_num == 0:
        return prompt
    elif retry_num == 1:
        # Remove examples and notes
        lines = [l for l in prompt.split("\n") if not l.strip().startswith("Note:") and "example" not in l.lower()]
        return "\n".join(lines)
    elif retry_num == 2:
        # Keep only the JSON schema part
        json_start = prompt.find("{")
        json_end = prompt.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return f"Return ONLY valid JSON matching this schema:\n{prompt[json_start:json_end]}"
        return prompt
    elif retry_num == 3:
        # Ultra-short: just the task + JSON
        first_line = prompt.split("\n")[0]
        json_start = prompt.find("{")
        json_end = prompt.rfind("}") + 1
        if json_start != -1:
            return f"{first_line}\nReturn JSON: {prompt[json_start:json_end]}"
        return first_line
    else:
        # Minimal: just ask for JSON
        return "Return valid JSON only. No explanation."


def _run_prompt_with_retry(images: list, prompt: str) -> str:
    """Run prompt with up to 7 retries, simplifying on each failure."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            current_prompt = _simplify_prompt(prompt, attempt)
            raw = _run_prompt(images, current_prompt)
            parsed = _extract_json(raw)
            if parsed is not None:
                return raw
            # JSON parse failed, retry with simpler prompt
            last_error = f"JSON parse failed on attempt {attempt + 1}"
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def _pass1_ocr(images: list) -> dict:
    """Pass 1: Extract visible text from all images with retry."""
    raw = _run_prompt_with_retry(images, PROMPT_OCR)
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        return parsed
    return {"text_blocks": [], "packaging_type": "unknown", "surfaces_visible": []}


def _pass2_structured(
    images: list, ocr_data: dict, user_text: str = ""
) -> dict[str, dict | list]:
    """Pass 2: Run 4 short prompts IN PARALLEL with OCR context."""
    ocr_text = format_ocr_for_prompt(ocr_data, max_chars=800)

    # User text takes priority — put it FIRST so VLM sees it before long OCR
    user_context = ""
    if user_text and user_text.strip():
        user_context = f"User-provided info:\n{user_text.strip()}\n\n"

    # Build all prompts first
    prompts_to_run = {}
    for prompt_name, prompt_template in PASS2_PROMPTS.items():
        if "{OCR_TEXT}" in prompt_template:
            combined = user_context + "OCR text:\n" + ocr_text if user_context else ocr_text
            prompts_to_run[prompt_name] = prompt_template.format(OCR_TEXT=combined)
        else:
            prompts_to_run[prompt_name] = prompt_template

    # Run all 4 prompts IN PARALLEL
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_prompt_with_retry, images, prompt): name
            for name, prompt in prompts_to_run.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                raw = future.result()
                parsed = _extract_json(raw)
                if parsed is not None:
                    results[name] = parsed
                else:
                    results[name] = {"_error": "JSON parse failed after retries"}
            except Exception as e:
                results[name] = {"_error": str(e)}

    return results


def _calculate_days_since_mfg(medicine_info: dict) -> int:
    """Calculate days since manufacturing date."""
    mfg_date_str = medicine_info.get("mfg_date")
    if mfg_date_str:
        mfg_date = parse_date(mfg_date_str)
        if mfg_date:
            return (datetime.now() - mfg_date).days
    return 60  # Default 60 days


def _calculate_shelf_life(medicine_info: dict) -> int:
    """Calculate shelf life in days from MFG and EXP dates."""
    mfg_date_str = medicine_info.get("mfg_date")
    exp_date_str = medicine_info.get("exp_date")
    if mfg_date_str and exp_date_str:
        mfg_date = parse_date(mfg_date_str)
        exp_date = parse_date(exp_date_str)
        if mfg_date and exp_date:
            return max(30, (exp_date - mfg_date).days)
    return 365  # Default 1 year


def analyze_image(images: list, user_text: str = "") -> AnalysisResult:
    """Run VLM analysis with Python fallbacks for complex calculations.
    
    Optimized with OpenBMB best practices:
    - Image preprocessing for medicine labels
    - Multi-image understanding
    - Efficient token usage (640 tokens per image)
    """
    result = AnalysisResult()

    if not images:
        result.errors.append("No images provided")
        return result

    # Preprocess images for medicine label analysis
    try:
        processed_images = _preprocess_images_for_medicine(images)
    except Exception as e:
        result.errors.append(f"Image preprocessing failed: {str(e)}")
        processed_images = images  # Fallback to original images

    # --- Pass 1: OCR extraction (with retry) ---
    try:
        ocr_data = _pass1_ocr(processed_images)
        result.ocr_data = ocr_data
        result.raw_responses["ocr"] = json.dumps(ocr_data, indent=2)
    except Exception as e:
        result.errors.append(f"OCR failed: {str(e)}")
        ocr_data = {"text_blocks": [], "packaging_type": "unknown", "surfaces_visible": []}

    if user_text and user_text.strip():
        result.raw_responses["user_text"] = user_text.strip()

    # --- Pass 2: Structured analysis (with retry) ---
    structured = _pass2_structured(processed_images, ocr_data, user_text=user_text)

    # Extract info
    if "info" in structured and "_error" not in structured["info"]:
        result.medicine_info = structured["info"]
        result.raw_responses["pass2_info"] = json.dumps(structured["info"], indent=2)
    else:
        result.errors.append(f"info: {structured.get('info', {}).get('_error', 'failed')}")

    # Extract spoilage
    if "spoilage" in structured and "_error" not in structured["spoilage"]:
        result.spoilage_assessment = structured["spoilage"]
        result.raw_responses["pass2_spoilage"] = json.dumps(structured["spoilage"], indent=2)
    else:
        result.errors.append(f"spoilage: {structured.get('spoilage', {}).get('_error', 'failed')}")

    # Extract bacteria
    if "bacteria" in structured and "_error" not in structured["bacteria"]:
        result.bacteria_estimate = structured["bacteria"]
        result.raw_responses["pass2_bacteria"] = json.dumps(structured["bacteria"], indent=2)
    else:
        result.errors.append(f"bacteria: {structured.get('bacteria', {}).get('_error', 'failed')}")

    # Extract chemicals
    if "chemicals" in structured and "_error" not in structured["chemicals"]:
        chem_data = structured["chemicals"]
        if isinstance(chem_data, dict) and "chemicals" in chem_data:
            result.chemicals = chem_data["chemicals"]
        elif isinstance(chem_data, list):
            result.chemicals = chem_data
        result.raw_responses["pass2_chemicals"] = json.dumps(chem_data, indent=2)
    else:
        result.errors.append(f"chemicals: {structured.get('chemicals', {}).get('_error', 'failed')}")

    # --- Python fallback calculations ---

    # Calculate bacteria growth curve
    try:
        ingredients = result.medicine_info.get("ingredients", [])
        preservatives = result.bacteria_estimate.get("preservatives_found", [])
        spoilage_level = result.spoilage_assessment.get("spoilage_level", 0)
        days_since_mfg = _calculate_days_since_mfg(result.medicine_info)
        shelf_life = _calculate_shelf_life(result.medicine_info)

        result.bacteria_growth_curve = calculate_theoretical_growth(
            ingredients=ingredients,
            preservatives=preservatives,
            shelf_life_days=shelf_life,
            days_since_mfg=days_since_mfg,
            spoilage_level=spoilage_level,
        )
        result.raw_responses["python_bacteria_growth"] = json.dumps(
            result.bacteria_growth_curve, indent=2
        )
    except Exception as e:
        result.errors.append(f"Bacteria growth calc: {str(e)}")

    # Calculate color analysis
    try:
        result.color_analysis = estimate_color_from_spoilage(result.spoilage_assessment)
        result.raw_responses["python_color_analysis"] = json.dumps(
            result.color_analysis, indent=2
        )
    except Exception as e:
        result.errors.append(f"Color analysis calc: {str(e)}")

    # Calculate dynamic expiry
    try:
        mfg_date = parse_date(result.medicine_info.get("mfg_date"))
        exp_date = parse_date(result.medicine_info.get("exp_date"))
        preservatives = result.bacteria_estimate.get("preservatives_found", [])
        color_deviation = result.color_analysis.get("color_deviation", 0.0)

        result.dynamic_expiry = calculate_dynamic_expiry(
            mfg_date=mfg_date,
            exp_date=exp_date,
            spoilage_assessment=result.spoilage_assessment,
            color_deviation=color_deviation,
            preservatives=preservatives,
        )
        result.raw_responses["python_dynamic_expiry"] = json.dumps(
            result.dynamic_expiry, indent=2
        )
    except Exception as e:
        result.errors.append(f"Dynamic expiry calc: {str(e)}")

    return result
