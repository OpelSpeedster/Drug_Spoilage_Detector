"""Test the full pipeline with a medicine image."""
import requests
import json
import base64
from PIL import Image
import io
import os

os.environ["MODAL_ENDPOINT_URL"] = "https://vishalsv2205--biochem-spoilage-detect-vlminference-analyze.modal.run"

# Create a test image (orange liquid like Jectocos Lipo)
img = Image.new("RGB", (200, 200), color=(255, 140, 50))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()

url = os.environ["MODAL_ENDPOINT_URL"]

# Test 1: OCR prompt (short)
ocr_prompt = """Read ALL visible text from this medicine packaging. Check front, sides, back, stickers, caps, and foil.

For each text block, return:
- text: exact text as shown
- location: front|side_panel|back|sticker|cap|foil|other
- source_image: photo number (1, 2, 3...)

Also identify:
- packaging_type: bottle|box|strip|blister|tube|sachet|other
- surfaces_visible: list of visible surfaces

Return ONLY valid JSON:
{"text_blocks": [{"text": "exact text", "location": "sticker", "source_image": 1}], "packaging_type": "bottle", "surfaces_visible": ["front", "sticker"]}
"""

print("=== Test 1: OCR Prompt ===")
try:
    resp = requests.post(url, params={"prompt": ocr_prompt}, json=[b64], timeout=120)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        response = data.get("response", "")
        print(f"Response length: {len(response)} chars")
        print(f"Response preview: {response[:300]}")
    else:
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Info prompt (short)
info_prompt = """Using the OCR text below, extract medicine information.

Return ONLY valid JSON:
{"name": "product name", "manufacturer": "company", "mfg_date": "date string or null", "exp_date": "date string or null", "batch_no": "batch number or null", "dosage_form": "syrup|tablet|capsule|other", "ingredients": ["list"], "storage_conditions": "text or null"}

If a field is not found, use null. Do not guess.

OCR text:
[photo 1, sticker] B.No: NJL25004 MFG: SEP 2025 EXP: FEB 2027
[photo 1, front] Jectocos Lipo - Iron, Folic Acid and Vitamin B12
"""

print("\n=== Test 2: Info Prompt ===")
try:
    resp = requests.post(url, params={"prompt": info_prompt}, json=[b64], timeout=120)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        response = data.get("response", "")
        print(f"Response length: {len(response)} chars")
        print(f"Response preview: {response[:300]}")
    else:
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Spoilage prompt (short)
spoilage_prompt = """Look at this medicine image. Assess visual spoilage.

Return ONLY valid JSON:
{"discoloration": true/false, "cloudiness": true/false, "sediment": true/false, "seal_intact": true/false, "label_damage": true/false, "spoilage_level": 0-100, "indicators": ["list"]}

Note: Suspensions (like iron syrup) are NATURALLY opaque. Do not flag that as cloudiness.
"""

print("\n=== Test 3: Spoilage Prompt ===")
try:
    resp = requests.post(url, params={"prompt": spoilage_prompt}, json=[b64], timeout=120)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        response = data.get("response", "")
        print(f"Response length: {len(response)} chars")
        print(f"Response preview: {response[:300]}")
    else:
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
