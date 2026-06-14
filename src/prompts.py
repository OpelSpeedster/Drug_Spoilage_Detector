"""VLM prompt templates for medicine spoilage detection.

Architecture:
  Pass 1 (PROMPT_OCR): Read visible text from all surfaces.
  Pass 2: Separate prompts for info, spoilage, bacteria, chemicals.

All prompts are kept SHORT (<400 tokens) to ensure VLM returns complete JSON.
Complex calculations (bacteria growth, dynamic expiry) are done in Python.
"""


def format_ocr_for_prompt(ocr_data: dict, max_chars: int = 1500) -> str:
    """Format OCR extraction results into a readable string for Pass 2 prompts.
    
    Truncates to max_chars to prevent VLM response truncation.
    """
    if not ocr_data or "text_blocks" not in ocr_data:
        return "(no OCR text available)"

    lines = []
    total = 0
    for block in ocr_data["text_blocks"]:
        loc = block.get("location", "unknown")
        text = block.get("text", "")
        source = block.get("source_image", "?")
        line = f"[photo {source}, {loc}] {text}"
        if total + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines) if lines else "(no OCR text available)"


# --- Pass 1: OCR Extraction (SHORT) ---

PROMPT_OCR = """Read ALL visible text from this medicine packaging. Check front, sides, back, stickers, caps, and foil.

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


# --- Pass 2: Info Extraction ---

PROMPT_INFO = """Look at this medicine image AND the OCR text below. Extract medicine information.

Return ONLY valid JSON:
{{"name": "product name", "manufacturer": "company", "mfg_date": "date string or null", "exp_date": "date string or null", "batch_no": "batch number or null", "dosage_form": "syrup|tablet|capsule|other", "ingredients": ["list"], "storage_conditions": "text or null"}}

If a field is not found in either the image or OCR text, use null. Do not guess.

IMPORTANT: If no expiry or manufacturing date is visible anywhere (label removed,
sticker missing, date rubbed off), set both mfg_date and exp_date to null.
Do NOT guess dates. The app will use visual analysis as fallback.

OCR text:
{OCR_TEXT}
"""


# --- Pass 2: Spoilage Assessment ---

PROMPT_SPOILAGE = """Look at this medicine image. Assess visual spoilage.

Return ONLY valid JSON:
{"discoloration": true/false, "cloudiness": true/false, "sediment": true/false, "seal_intact": true/false, "label_damage": true/false, "spoilage_level": 0-100, "indicators": []}

indicators: list any specific observations like "rust color", "particles visible", "seal broken", etc. Use empty list if none.

Note: Suspensions (like iron syrup) are NATURALLY opaque. Do not flag that as cloudiness.
"""


# --- Pass 2: Bacteria Risk ---

PROMPT_BACTERIA = """Look at this medicine image. Assess bacterial contamination risk based on what you SEE.

Check for: cloudiness, sediment, discoloration, floating particles, seal condition, storage signs.

Return ONLY valid JSON:
{"growth_level": 0-100, "preservatives_found": ["list from ingredients if visible"], "reasoning": "brief reason based on visual evidence only"}

Do NOT assume packaging damage if the seal appears intact.
"""


# --- Pass 2: Chemicals ---

PROMPT_CHEMICALS = """Look at this medicine image AND the OCR text below. List all chemical ingredients with quantities.

Return ONLY valid JSON:
{{"chemicals": [{{"name": "chemical name", "quantity": "amount with unit (e.g. 5mg, 2.5ml, 100mcg) or null", "category": "active_ingredient|preservative|solvent|other", "risk_level": "safe|caution|danger"}}]}}

If OCR text is incomplete, read directly from the image labels.

OCR text:
{OCR_TEXT}
"""


# --- Prompt lookup for Pass 2 ---

PASS2_PROMPTS = {
    "info": PROMPT_INFO,
    "spoilage": PROMPT_SPOILAGE,
    "bacteria": PROMPT_BACTERIA,
    "chemicals": PROMPT_CHEMICALS,
}
