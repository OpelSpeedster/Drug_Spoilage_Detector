"""Drug Spoilage Detector — Main Gradio App.

Detects medicine spoilage from images using MiniCPM-V 2.6 via Modal.
Upload photos of a medicine to detect spoilage, visualize chemical
composition, estimate bacteria growth, and compare expiry dates.

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

from src.engine import analyze_image
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

    risk_map = {"safe": 10, "caution": 50, "danger": 90, "unknown": 20}
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


def process_images(images, user_text: str):
    """Main analysis pipeline — analyzes all uploaded images as one medicine.

    Gallery supports upload, webcam (multiple shots), and clipboard.
    Returns 11 outputs: info, verdict, date, 6 charts, radar, raw responses.
    """
    if not images:
        placeholder = "### Medicine\n\nUpload photos or take photos with camera, then click Analyze."
        empty_chart = gr.update()
        return [
            gr.Markdown(placeholder), gr.Markdown(""), gr.Markdown(""),
            empty_chart, empty_chart, empty_chart,
            empty_chart, empty_chart, empty_chart,
            empty_chart, gr.Markdown(""),
        ]

    processed_images = preprocess_uploaded_images(images)
    result = analyze_image(processed_images, user_text=user_text)
    return extract_drug_outputs(result)


# --- Build Gradio UI ---
THEME = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.header-banner {
    background: linear-gradient(135deg, #059669 0%, #0d9488 50%, #0891b2 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 16px;
}
.header-banner h1 {
    color: white !important;
    font-size: 1.8em !important;
    margin: 0 !important;
}
.header-banner p {
    color: rgba(255,255,255,0.85) !important;
    margin: 4px 0 0 0 !important;
    font-size: 0.95em !important;
}
.verdict-card {
    padding: 20px 24px;
    border-radius: 10px;
    border-left: 5px solid;
    margin: 12px 0;
}
.verdict-card .verdict-label {
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.7;
    margin-bottom: 4px;
}
.verdict-card .verdict-text {
    font-size: 1.5em;
    font-weight: 700;
    margin: 0;
}
.verdict-card .verdict-score {
    font-size: 0.95em;
    margin-top: 6px;
}
.tip-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88em;
    line-height: 1.5;
    color: #e2e8f0;
    margin: 8px 0;
}
.tip-box strong { color: #f8fafc; }
.tip-box ul { margin: 6px 0 0 16px; padding: 0; }
.tip-box li { margin: 2px 0; color: #cbd5e1; }
.analyze-btn {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    padding: 10px 0 !important;
    border-radius: 8px !important;
    transition: opacity 0.2s !important;
}
.analyze-btn:hover { opacity: 0.9 !important; }
.chart-section { margin-top: 8px; }
.section-header {
    font-size: 1.1em;
    font-weight: 600;
    color: #1e293b;
    margin: 20px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}
gradio-container { max-width: 1200px !important; }
footer { margin-top: 16px; }
"""

with gr.Blocks(title="Drug Spoilage Detector") as demo:
    # --- Header Banner ---
    gr.HTML("""
    <div class="header-banner">
        <h1>Drug Spoilage Detector</h1>
        <p>Powered by MiniCPM-V 2.6 INT4 (8B params) on Modal</p>
    </div>
    """)

    with gr.Sidebar(position="left", open=True):
        gr.Markdown("## Upload Medicine")

        image_input = gr.Gallery(
            label="Medicine Photos (upload or take with camera)",
            columns=3,
            height=280,
            file_types=["image"],
            type="pil",
            sources=["upload", "webcam", "clipboard"],
        )

        gr.HTML("""
        <div class="tip-box">
            <strong>Tips for best results:</strong>
            <ul>
                <li>Upload 2-4 photos from different angles</li>
                <li>Ensure text on labels is clear and readable</li>
                <li>Works with or without expiry dates</li>
                <li>Supports Ayurvedic & herbal medicines</li>
            </ul>
        </div>
        """)

        user_text_input = gr.Textbox(
            label="Additional Info (optional)",
            placeholder="Medicine name, what you see on label, storage conditions...",
            lines=2,
        )
        analyze_btn = gr.Button(
            "Analyze",
            variant="primary",
            size="lg",
            elem_classes=["analyze-btn"],
        )

    # --- Main Content ---
    # Verdict banner
    verdict_md = gr.Markdown("")

    # Extracted info
    info_md = gr.Markdown(
        """
        ### Extracted Information

        Upload photos and click **Analyze** to see medicine details.
        """
    )

    # Tabs for chart groups
    with gr.Tabs() as tabs:
        with gr.Tab("Overview"):
            date_md = gr.Markdown("")

        with gr.Tab("Chemical"):
            chem_plot = gr.Plot(label="Chemical Composition")

        with gr.Tab("Bacteria"):
            with gr.Row():
                gauge_plot = gr.Plot(label="Bacteria Growth")
                radar_plot = gr.Plot(label="Risk Assessment")
            growth_plot = gr.Plot(label="Bacteria Growth Curve")

        with gr.Tab("Visual"):
            timeline_plot = gr.Plot(label="Spoilage Timeline")
            color_plot = gr.Plot(label="Color Degradation")

        with gr.Tab("Expiry"):
            dynamic_plot = gr.Plot(label="Static vs Dynamic Expiry")

    # Raw Responses
    with gr.Accordion("Raw VLM Responses", open=False):
        raw_md = gr.Markdown()

    # --- Footer ---
    gr.Markdown(
        """
        ---
        Built for [Build Small Hackathon](https://huggingface.co/build-small-hackathon) — Track 1: Backyard AI.
        Model: `openbmb/MiniCPM-V-2_6-int4` (8B params) served via Modal. Inference runs on Modal GPUs.
        """
    )

    # Wire up the analysis
    all_outputs = [
        info_md, verdict_md, date_md,
        chem_plot, timeline_plot, gauge_plot,
        growth_plot, color_plot, dynamic_plot,
        radar_plot, raw_md,
    ]

    analyze_btn.click(
        fn=process_images,
        inputs=[image_input, user_text_input],
        outputs=all_outputs,
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
