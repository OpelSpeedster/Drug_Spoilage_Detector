"""Biochem Spoilage Detect — Main Gradio App.

Detects medicine spoilage from images using MiniCPM-V 2.6 via Modal.
Supports multi-drug detection — upload photos of multiple medicines
and get separate analysis for each, displayed side-by-side.

Optimized with OpenBMB best practices:
- Image preprocessing for medicine labels
- Multi-image understanding
- Efficient token usage
"""

from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image

from src.engine import analyze_image, detect_drugs
from src.utils import (
    parse_date,
    calculate_spoilage_score,
    get_spoilage_verdict,
    get_verdict_color,
    enrich_chemicals,
)
from src.visualization import (
    create_chemical_bar_chart,
    create_spoilage_timeline,
    create_bacteria_gauge,
    create_bacteria_growth_curve,
    create_color_degradation_timeline,
    create_dynamic_expiry_comparison,
    create_risk_radar,
)


MAX_DRUGS = 3


def preprocess_uploaded_images(images) -> list:
    """Preprocess uploaded images for optimal VLM analysis."""
    if not images:
        return []

    processed = []
    for img in images:
        if isinstance(img, tuple):
            img = img[0]
        if not isinstance(img, Image.Image):
            try:
                import numpy as np
                img = Image.fromarray(np.array(img))
            except:
                continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        max_size = 1344
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        from PIL import ImageEnhance, ImageFilter
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img = img.filter(ImageFilter.SHARPEN)

        processed.append(img)

    return processed


def extract_drug_outputs(result, drug_name="Medicine") -> list:
    """Extract 11 UI outputs from an AnalysisResult for one drug."""
    info = result.medicine_info
    name = info.get("name", "Not detected")
    manufacturer = info.get("manufacturer", "Not detected")
    mfg_date_str = info.get("mfg_date", "")
    exp_date_str = info.get("exp_date", "")
    batch_no = info.get("batch_no", "Not detected")
    dosage_form = info.get("dosage_form", "Not detected")
    ingredients = info.get("ingredients", [])
    storage = info.get("storage_conditions", "Not detected")

    mfg_date = parse_date(mfg_date_str) if mfg_date_str else None
    exp_date = parse_date(exp_date_str) if exp_date_str else None
    today = datetime.now()

    mfg_display = mfg_date.strftime("%d %b %Y") if mfg_date else (mfg_date_str or "Not detected")
    exp_display = exp_date.strftime("%d %b %Y") if exp_date else (exp_date_str or "Not detected")

    spoilage = result.spoilage_assessment
    visual_indicators = []
    if spoilage.get("discoloration"):
        visual_indicators.append("Discoloration")
    if spoilage.get("cloudiness"):
        visual_indicators.append("Cloudiness")
    if spoilage.get("sediment"):
        visual_indicators.append("Sediment")
    if not spoilage.get("seal_intact", True):
        visual_indicators.append("Seal broken")
    if spoilage.get("label_damage"):
        visual_indicators.append("Label damage")

    visual_level = spoilage.get("spoilage_level", 50)
    visual_str = ", ".join(visual_indicators) if visual_indicators else "None detected"

    bacteria = result.bacteria_estimate
    bacteria_level = bacteria.get("growth_level", 0)

    bacteria_growth_curve = result.bacteria_growth_curve
    color_analysis = result.color_analysis
    dynamic_expiry = result.dynamic_expiry
    color_deviation = color_analysis.get("color_deviation", 0.0)
    static_expiry_days = dynamic_expiry.get("shelf_life_days")
    dynamic_expiry_days = dynamic_expiry.get("days_until_dynamic_expiry")
    adjustment_factors = dynamic_expiry.get("adjustment_factors", {})

    chemicals = enrich_chemicals(result.chemicals)

    shelf_life_days = 365
    if mfg_date and exp_date:
        shelf_life_days = (exp_date - mfg_date).days

    spoilage_score = calculate_spoilage_score(
        visual_level=visual_level,
        bacteria_level=bacteria_level,
        days_until_expiry=(exp_date - today).days if exp_date else 365,
        shelf_life_days=shelf_life_days,
        color_deviation=color_deviation,
        dynamic_expiry_days=dynamic_expiry_days,
    )

    verdict = get_spoilage_verdict(spoilage_score)
    verdict_color = get_verdict_color(spoilage_score)

    # Date comparison (two modes)
    if exp_date:
        days_until_expiry = (exp_date - today).days
        if days_until_expiry < 0:
            date_verdict = f"EXPIRED {abs(days_until_expiry)} days ago"
        elif days_until_expiry < 30:
            date_verdict = f"Near expiry — only {days_until_expiry} days left"
        elif days_until_expiry < 90:
            date_verdict = f"Use soon — {days_until_expiry} days left"
        else:
            date_verdict = f"Safe — {days_until_expiry} days until expiry"

        if mfg_date:
            shelf_life_total = (exp_date - mfg_date).days
            spoilage_factor = spoilage_score / 100.0
            predicted_spoilage = mfg_date + timedelta(days=max(30, int(shelf_life_total * (1 - spoilage_factor))))
            predicted_spoilage_str = predicted_spoilage.strftime("%b %Y")
        else:
            predicted_spoilage = None
            predicted_spoilage_str = "N/A (no mfg date)"

        date_md = f"""### Date Comparison

| Metric | Value |
|--------|-------|
| **Static Expiry** | {exp_display} |
| **Days Remaining** | {days_until_expiry} |
| **Visual Estimate** | {predicted_spoilage_str} *(secondary)* |
| **Today** | {today.strftime("%d %b %Y")} |
| **Verdict** | {date_verdict} |
"""
    else:
        visual_days_remaining = max(0, int(365 * (1 - spoilage_score / 100)))
        if spoilage_score > 60:
            date_verdict = "Visually spoiled — do not use"
        elif spoilage_score > 30:
            date_verdict = f"Moderate spoilage — estimated {visual_days_remaining} days usable"
        else:
            date_verdict = f"Visually OK — estimated {visual_days_remaining} days usable"

        predicted_spoilage = today + timedelta(days=visual_days_remaining)
        predicted_spoilage_str = predicted_spoilage.strftime("%b %Y")
        days_until_expiry = visual_days_remaining

        date_md = f"""### Date Comparison

| Metric | Value |
|--------|-------|
| **Static Expiry** | Not found on label |
| **Visual Estimate** | {predicted_spoilage_str} *(primary — no label found)* |
| **Estimated Days Left** | ~{visual_days_remaining} days |
| **Today** | {today.strftime("%d %b %Y")} |
| **Verdict** | {date_verdict} |
"""

    info_md = f"""### {drug_name} — Extracted Information

| Field | Value |
|-------|-------|
| **Name** | {name} |
| **Manufacturer** | {manufacturer} |
| **Manufacturing Date** | {mfg_display} |
| **Expiry Date** | {exp_display} |
| **Batch No.** | {batch_no} |
| **Dosage Form** | {dosage_form} |
| **Storage** | {storage} |
| **Ingredients** | {', '.join(ingredients) if ingredients else 'Not detected'} |
"""

    verdict_md = f"""### Spoilage Assessment

<div style="background-color: {verdict_color}20; border-left: 4px solid {verdict_color}; padding: 12px; border-radius: 4px;">

**Status: <span style="color: {verdict_color}; font-size: 1.2em;">{verdict}</span>**

**Spoilage Score: {spoilage_score}/100**

</div>

| Indicator | Finding |
|-----------|---------|
| **Visual Signs** | {visual_str} |
| **Bacteria Growth** | {bacteria_level}/100 |
| **Visual Level** | {visual_level}/100 |
"""

    chemical_chart = create_chemical_bar_chart(chemicals)
    timeline_chart = create_spoilage_timeline(
        mfg_date=mfg_date, exp_date=exp_date,
        predicted_spoilage=predicted_spoilage if exp_date else predicted_spoilage,
        today=today,
    )
    gauge_chart = create_bacteria_gauge(bacteria_level)
    growth_curve_chart = create_bacteria_growth_curve(
        growth_curve=bacteria_growth_curve,
        current_day=bacteria_growth_curve.get("factors", {}).get("days_since_manufacturing", 0),
        critical_threshold_day=bacteria_growth_curve.get("critical_threshold_day", 200),
    )
    color_timeline_chart = create_color_degradation_timeline(
        color_analysis=color_analysis, shelf_life_days=shelf_life_days,
    )
    dynamic_expiry_chart = create_dynamic_expiry_comparison(
        static_expiry_days=static_expiry_days,
        dynamic_expiry_days=dynamic_expiry_days,
        adjustment_factors=adjustment_factors,
    )

    risk_map = {"safe": 10, "caution": 50, "danger": 90, "unknown": 50}
    avg_chemical_risk = (
        sum(risk_map.get(c.get("risk_level", "unknown"), 50) for c in chemicals) / len(chemicals)
        if chemicals else 0
    )
    date_score = 0
    if days_until_expiry is not None and shelf_life_days > 0:
        date_score = int(max(0, 1 - (days_until_expiry / shelf_life_days)) * 100)

    radar_chart = create_risk_radar(
        visual_score=visual_level, bacteria_score=bacteria_level,
        date_score=date_score, chemical_risk=avg_chemical_risk,
    )

    raw_md = f"### {drug_name} — Raw VLM Responses\n\n"
    for prompt_name, raw_text in result.raw_responses.items():
        raw_md += f"**{prompt_name}:**\n```\n{raw_text}\n```\n\n"
    if result.errors:
        raw_md += "### Errors\n"
        for err in result.errors:
            raw_md += f"- {err}\n"

    return [
        info_md, verdict_md, date_md,
        chemical_chart, timeline_chart, gauge_chart,
        growth_curve_chart, color_timeline_chart, dynamic_expiry_chart,
        radar_chart, raw_md,
    ]


def empty_drug_outputs(drug_name="No image uploaded") -> list:
    """Return 11 empty/placeholder outputs for an inactive drug slot."""
    placeholder = f"### {drug_name}\n\nUpload photos and click Analyze to see results."
    empty_chart = gr.update()
    return [
        gr.Markdown(placeholder),
        gr.Markdown(""),
        gr.Markdown(""),
        empty_chart, empty_chart, empty_chart,
        empty_chart, empty_chart, empty_chart,
        empty_chart, gr.Markdown(""),
    ]


def process_images(images, user_text: str):
    """Main analysis pipeline — detects multiple drugs and analyzes each.

    Returns 36 outputs: 11 per drug × 3 slots + 3 column visibility toggles.
    """
    # 36 = 11 outputs × 3 drugs + 3 visibility flags
    if not images:
        outputs = []
        for i in range(MAX_DRUGS):
            outputs.extend(empty_drug_outputs(f"Drug {i+1}"))
            outputs.append(gr.update(visible=False))
        return outputs

    processed_images = preprocess_uploaded_images(images)

    # Step 1: Detect distinct drugs
    detection = detect_drugs(processed_images)
    drug_count = min(detection["drug_count"], MAX_DRUGS)
    drugs = detection["drugs"][:MAX_DRUGS]

    # Step 2: Analyze each drug
    outputs = []
    for i in range(MAX_DRUGS):
        if i < drug_count and drugs[i]:
            drug = drugs[i]
            drug_name = drug.get("name", f"Drug {i+1}")
            photo_indices = drug.get("photo_indices", [1])

            # Get images for this drug (1-based indices)
            drug_images = []
            for idx in photo_indices:
                if 1 <= idx <= len(processed_images):
                    drug_images.append(processed_images[idx - 1])

            if not drug_images:
                drug_images = processed_images  # Fallback: use all images

            result = analyze_image(drug_images, user_text=user_text)
            outputs.extend(extract_drug_outputs(result, drug_name))
            outputs.append(gr.update(visible=True))
        else:
            outputs.extend(empty_drug_outputs(f"Drug {i+1}"))
            outputs.append(gr.update(visible=False))

    return outputs


# --- Build Gradio UI ---
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="pink",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.drug-column { border-left: 3px solid #3366CC; padding-left: 12px; }
"""

with gr.Blocks(title="Biochem Spoilage Detect", css=CSS) as demo:
    gr.Markdown(
        """
        # Biochem Spoilage Detect
        ### Powered by MiniCPM-V 2.6 INT4 (8B params) on Modal

        Upload photos of one or more medicines to detect spoilage, visualize chemical
        composition, estimate bacteria growth, and compare expiry dates.
        **Multiple drugs are automatically detected and analyzed separately.**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Gallery(
                label="Upload Medicine Photos (front, side, back, sticker...)",
                columns=3,
                height=300,
                file_types=["image"],
                type="pil",
                sources=["upload", "webcam", "clipboard"],
            )
            gr.Markdown(
                """
                **Tips for best results:**
                - Upload 2-4 photos per medicine from different angles
                - Ensure text on labels is clear and readable
                - You can upload photos of multiple drugs — they'll be detected automatically
                - Images are automatically optimized for analysis
                """
            )
            user_text_input = gr.Textbox(
                label="Additional Info (optional)",
                placeholder="Type any details: medicine names, what you see on labels, storage conditions, etc.",
                lines=3,
            )
            analyze_btn = gr.Button(
                "Analyze Photos",
                variant="primary",
                size="lg",
            )

    # --- Drug 1 Results ---
    with gr.Row():
        drug1_col = gr.Column(visible=True, elem_classes=["drug-column"])
        with drug1_col:
            drug1_info = gr.Markdown("### Drug 1\n\nUpload photos and click Analyze...")
            drug1_verdict = gr.Markdown("")
            drug1_date = gr.Markdown("")
        drug2_col = gr.Column(visible=False, elem_classes=["drug-column"])
        with drug2_col:
            drug2_info = gr.Markdown("### Drug 2\n\nUpload photos and click Analyze...")
            drug2_verdict = gr.Markdown("")
            drug2_date = gr.Markdown("")
        drug3_col = gr.Column(visible=False, elem_classes=["drug-column"])
        with drug3_col:
            drug3_info = gr.Markdown("### Drug 3\n\nUpload photos and click Analyze...")
            drug3_verdict = gr.Markdown("")
            drug3_date = gr.Markdown("")

    # --- Drug 1 Charts ---
    with gr.Row():
        drug1_chem = gr.Plot(label="Chemical Composition")
        drug1_timeline = gr.Plot(label="Spoilage Timeline")
    with gr.Row():
        drug1_gauge = gr.Plot(label="Bacteria Growth")
        drug1_radar = gr.Plot(label="Risk Radar")
    with gr.Row():
        drug1_growth = gr.Plot(label="Bacteria Growth Curve")
        drug1_color = gr.Plot(label="Color Degradation")
    with gr.Row():
        drug1_dynamic = gr.Plot(label="Static vs Dynamic Expiry")

    # --- Drug 2 Charts ---
    with gr.Row(visible=False) as drug2_charts_row:
        drug2_chem = gr.Plot(label="Chemical Composition")
        drug2_timeline = gr.Plot(label="Spoilage Timeline")
    with gr.Row(visible=False) as drug2_charts_row2:
        drug2_gauge = gr.Plot(label="Bacteria Growth")
        drug2_radar = gr.Plot(label="Risk Radar")
    with gr.Row(visible=False) as drug2_charts_row3:
        drug2_growth = gr.Plot(label="Bacteria Growth Curve")
        drug2_color = gr.Plot(label="Color Degradation")
    with gr.Row(visible=False) as drug2_charts_row4:
        drug2_dynamic = gr.Plot(label="Static vs Dynamic Expiry")

    # --- Drug 3 Charts ---
    with gr.Row(visible=False) as drug3_charts_row:
        drug3_chem = gr.Plot(label="Chemical Composition")
        drug3_timeline = gr.Plot(label="Spoilage Timeline")
    with gr.Row(visible=False) as drug3_charts_row2:
        drug3_gauge = gr.Plot(label="Bacteria Growth")
        drug3_radar = gr.Plot(label="Risk Radar")
    with gr.Row(visible=False) as drug3_charts_row3:
        drug3_growth = gr.Plot(label="Bacteria Growth Curve")
        drug3_color = gr.Plot(label="Color Degradation")
    with gr.Row(visible=False) as drug3_charts_row4:
        drug3_dynamic = gr.Plot(label="Static vs Dynamic Expiry")

    # --- Raw Responses ---
    with gr.Accordion("Raw VLM Responses", open=False):
        drug1_raw = gr.Markdown()
        drug2_raw = gr.Markdown()
        drug3_raw = gr.Markdown()

    # Wire up the analysis
    all_outputs = [
        # Drug 1 (always visible)
        drug1_info, drug1_verdict, drug1_date,
        drug1_chem, drug1_timeline, drug1_gauge,
        drug1_growth, drug1_color, drug1_dynamic,
        drug1_radar, drug1_raw,
        drug1_col,
        # Drug 2
        drug2_info, drug2_verdict, drug2_date,
        drug2_chem, drug2_timeline, drug2_gauge,
        drug2_growth, drug2_color, drug2_dynamic,
        drug2_radar, drug2_raw,
        drug2_col,
        # Drug 3
        drug3_info, drug3_verdict, drug3_date,
        drug3_chem, drug3_timeline, drug3_gauge,
        drug3_growth, drug3_color, drug3_dynamic,
        drug3_radar, drug3_raw,
        drug3_col,
    ]

    analyze_btn.click(
        fn=process_images,
        inputs=[image_input, user_text_input],
        outputs=all_outputs,
    )

    gr.Markdown(
        """
        ---
        Built for [Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: Backyard AI.
        Model: `openbmb/MiniCPM-V-2_6-int4` (8B params) served via Modal. Inference runs on Modal GPUs.
        """
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
