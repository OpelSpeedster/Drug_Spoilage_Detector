"""Date parsing, chemical reference DB, and spoilage scoring utilities."""

from datetime import datetime, timedelta
from dateutil import parser as date_parser
import math
import re

# --- Chemical Reference Database ---
CHEMICAL_DB = {
    "paracetamol": {"category": "active_ingredient", "risk_level": "safe"},
    "acetaminophen": {"category": "active_ingredient", "risk_level": "safe"},
    "ibuprofen": {"category": "active_ingredient", "risk_level": "safe"},
    "amoxicillin": {"category": "active_ingredient", "risk_level": "safe"},
    "amoxicilline": {"category": "active_ingredient", "risk_level": "safe"},
    "cetirizine": {"category": "active_ingredient", "risk_level": "safe"},
    "loratadine": {"category": "active_ingredient", "risk_level": "safe"},
    "metformin": {"category": "active_ingredient", "risk_level": "safe"},
    "omeprazole": {"category": "active_ingredient", "risk_level": "safe"},
    "pantoprazole": {"category": "active_ingredient", "risk_level": "safe"},
    "azithromycin": {"category": "active_ingredient", "risk_level": "safe"},
    "ciprofloxacin": {"category": "active_ingredient", "risk_level": "safe"},
    "doxycycline": {"category": "active_ingredient", "risk_level": "safe"},
    "metronidazole": {"category": "active_ingredient", "risk_level": "safe"},
    "diclofenac": {"category": "active_ingredient", "risk_level": "safe"},
    "naproxen": {"category": "active_ingredient", "risk_level": "safe"},
    "aspirin": {"category": "active_ingredient", "risk_level": "safe"},
    "salbutamol": {"category": "active_ingredient", "risk_level": "safe"},
    "prednisolone": {"category": "active_ingredient", "risk_level": "safe"},
    "dexamethasone": {"category": "active_ingredient", "risk_level": "safe"},
    "montelukast": {"category": "active_ingredient", "risk_level": "safe"},
    "levocetirizine": {"category": "active_ingredient", "risk_level": "safe"},
    "hydroxyzine": {"category": "active_ingredient", "risk_level": "caution"},
    "promethazine": {"category": "active_ingredient", "risk_level": "caution"},
    "chlorpheniramine": {"category": "active_ingredient", "risk_level": "safe"},
    "phenylephrine": {"category": "active_ingredient", "risk_level": "caution"},
    "pseudoephedrine": {"category": "active_ingredient", "risk_level": "caution"},
    "guaifenesin": {"category": "active_ingredient", "risk_level": "safe"},
    "ambroxol": {"category": "active_ingredient", "risk_level": "safe"},
    "bromhexine": {"category": "active_ingredient", "risk_level": "safe"},
    "terbutaline": {"category": "active_ingredient", "risk_level": "safe"},
    "theophylline": {"category": "active_ingredient", "risk_level": "caution"},
    "ranitidine": {"category": "active_ingredient", "risk_level": "caution"},
    "famotidine": {"category": "active_ingredient", "risk_level": "safe"},
    "loperamide": {"category": "active_ingredient", "risk_level": "safe"},
    "ondansetron": {"category": "active_ingredient", "risk_level": "safe"},
    "domperidone": {"category": "active_ingredient", "risk_level": "caution"},
    "metoclopramide": {"category": "active_ingredient", "risk_level": "caution"},
    "sucralfate": {"category": "active_ingredient", "risk_level": "safe"},
    "misoprostol": {"category": "active_ingredient", "risk_level": "danger"},
    "sodium_benzoate": {"category": "preservative", "risk_level": "caution"},
    "potassium_sorbate": {"category": "preservative", "risk_level": "safe"},
    "methylparaben": {"category": "preservative", "risk_level": "caution"},
    "propylparaben": {"category": "preservative", "risk_level": "caution"},
    "benzoic_acid": {"category": "preservative", "risk_level": "caution"},
    "sorbic_acid": {"category": "preservative", "risk_level": "safe"},
    "edta": {"category": "preservative", "risk_level": "safe"},
    "disodium_edta": {"category": "preservative", "risk_level": "safe"},
    "glycerin": {"category": "solvent", "risk_level": "safe"},
    "glycerol": {"category": "solvent", "risk_level": "safe"},
    "propylene_glycol": {"category": "solvent", "risk_level": "safe"},
    "sorbitol": {"category": "solvent", "risk_level": "safe"},
    "mannitol": {"category": "solvent", "risk_level": "safe"},
    "ethanol": {"category": "solvent", "risk_level": "caution"},
    "alcohol": {"category": "solvent", "risk_level": "caution"},
    "water": {"category": "solvent", "risk_level": "safe"},
    "purified_water": {"category": "solvent", "risk_level": "safe"},
    "sodium_chloride": {"category": "excipient", "risk_level": "safe"},
    "calcium_carbonate": {"category": "excipient", "risk_level": "safe"},
    "microcrystalline_cellulose": {"category": "binder", "risk_level": "safe"},
    "methylcellulose": {"category": "binder", "risk_level": "safe"},
    "hydroxypropyl_methylcellulose": {"category": "binder", "risk_level": "safe"},
    "povidone": {"category": "binder", "risk_level": "safe"},
    "croscarmellose_sodium": {"category": "disintegrant", "risk_level": "safe"},
    "starch": {"category": "filler", "risk_level": "safe"},
    "lactose": {"category": "filler", "risk_level": "safe"},
    "magnesium_stearate": {"category": "lubricant", "risk_level": "safe"},
    "talc": {"category": "lubricant", "risk_level": "safe"},
    "titanium_dioxide": {"category": "colorant", "risk_level": "caution"},
    "sunset_yellow": {"category": "colorant", "risk_level": "caution"},
    "tartrazine": {"category": "colorant", "risk_level": "caution"},
    "brilliant_blue": {"category": "colorant", "risk_level": "safe"},
    "allura_red": {"category": "colorant", "risk_level": "caution"},
    "carmine": {"category": "colorant", "risk_level": "safe"},
    "sucrose": {"category": "sweetener", "risk_level": "safe"},
    "sugar": {"category": "sweetener", "risk_level": "safe"},
    "aspartame": {"category": "sweetener", "risk_level": "caution"},
    "saccharin": {"category": "sweetener", "risk_level": "caution"},
    "xylitol": {"category": "sweetener", "risk_level": "safe"},
    "sodium_saccharin": {"category": "sweetener", "risk_level": "caution"},
    "citric_acid": {"category": "acidifier", "risk_level": "safe"},
    "sodium_citrate": {"category": "buffer", "risk_level": "safe"},
    "tartaric_acid": {"category": "acidifier", "risk_level": "safe"},
    "phosphoric_acid": {"category": "acidifier", "risk_level": "caution"},
    "sodium_hydroxide": {"category": "ph_adjuster", "risk_level": "caution"},
    "hydrochloric_acid": {"category": "ph_adjuster", "risk_level": "danger"},
    "banana_flavor": {"category": "flavoring", "risk_level": "safe"},
    "orange_flavor": {"category": "flavoring", "risk_level": "safe"},
    "vanilla_flavor": {"category": "flavoring", "risk_level": "safe"},
    "peppermint_oil": {"category": "flavoring", "risk_level": "safe"},
    "mint": {"category": "flavoring", "risk_level": "safe"},
}


def parse_date(date_str: str) -> datetime | None:
    """Parse date from various formats commonly found on medicine packaging."""
    if not date_str:
        return None
    date_str = str(date_str).strip()
    # Normalize dot-separated dates: "SEP.2025" → "SEP 2025", "SEP.25" → "SEP 25"
    date_str = re.sub(r'(\w+)\.(\d)', r'\1 \2', date_str)

    # Try common patterns first
    patterns = [
        r"(\d{2})/(\d{2})/(\d{4})",      # DD/MM/YYYY
        r"(\d{2})/(\d{2})/(\d{2})",        # DD/MM/YY
        r"(\d{4})-(\d{2})-(\d{2})",        # YYYY-MM-DD
        r"(\d{2})-(\d{2})-(\d{4})",        # DD-MM-YYYY
        r"(\d{2})\.(\d{2})\.(\d{4})",      # DD.MM.YYYY
        r"(\d{2})/(\d{4})",                 # MM/YYYY
        r"(\d{2})-(\d{4})",                 # MM-YYYY
        r"(\w+)\s+(\d{4})",                 # Month YYYY (e.g., "Jan 2025") — must come before YYYY-only
        r"(\d{4})",                         # YYYY only
        r"(\d{1,2})\s+(\w+)\s+(\d{4})",   # DD Month YYYY
    ]

    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 3 and len(groups[2]) == 4:
                    # DD/MM/YYYY or DD-MM-YYYY
                    day, month, year = groups
                    return datetime(int(year), int(month), int(day))
                elif len(groups) == 3 and len(groups[0]) == 4:
                    # YYYY-MM-DD
                    year, month, day = groups
                    return datetime(int(year), int(month), int(day))
                elif len(groups) == 3 and len(groups[0]) <= 2:
                    # DD Month YYYY
                    day, month_str, year = groups
                    return date_parser.parse(f"{month_str} {day} {year}")
                elif len(groups) == 2 and len(groups[1]) == 4:
                    # MM/YYYY or Month YYYY
                    return date_parser.parse(f"{groups[0]} {groups[1]}")
                elif len(groups) == 1 and len(groups[0]) == 4:
                    return datetime(int(groups[0]), 1, 1)
            except (ValueError, TypeError):
                continue

    # Fallback to dateutil parser
    try:
        return date_parser.parse(date_str, dayfirst=True)
    except (ValueError, TypeError):
        return None


def calculate_spoilage_score(
    visual_level: int,
    bacteria_level: int,
    days_until_expiry: int,
    shelf_life_days: int,
    color_deviation: float = 0.0,
    dynamic_expiry_days: int = None,
) -> int:
    """Calculate spoilage score (0-100) from weighted factors.

    Weights: visual (35%), bacteria (25%), date (20%), color (10%), dynamic expiry (10%)
    """
    # Visual component (0-100)
    visual_score = min(max(visual_level, 0), 100)

    # Bacteria component (0-100)
    bacteria_score = min(max(bacteria_level, 0), 100)

    # Date component: closer to expiry = higher score
    if shelf_life_days > 0 and days_until_expiry is not None:
        date_ratio = max(0, 1 - (days_until_expiry / shelf_life_days))
        date_score = int(date_ratio * 100)
    else:
        date_score = 50  # unknown if no dates

    # Color component (0-100)
    color_score = int(min(max(color_deviation, 0), 1.0) * 100)

    # Dynamic expiry component (0-100)
    if dynamic_expiry_days is not None and dynamic_expiry_days >= 0:
        # Higher score if dynamic expiry is closer (more spoiled)
        dynamic_score = max(0, 100 - int(dynamic_expiry_days / 3.65))
    else:
        dynamic_score = 50  # unknown

    # Weighted sum
    score = int(
        visual_score * 0.35 +
        bacteria_score * 0.25 +
        date_score * 0.20 +
        color_score * 0.10 +
        dynamic_score * 0.10
    )
    return min(max(score, 0), 100)


def get_spoilage_verdict(score: int) -> str:
    """Return verdict string from spoilage score."""
    if score > 60:
        return "SPOILED"
    elif score > 30:
        return "WARNING"
    else:
        return "SAFE"


def get_verdict_color(score: int) -> str:
    """Return hex color for verdict."""
    if score > 60:
        return "#FF4444"  # Red
    elif score > 30:
        return "#FFAA00"  # Amber
    else:
        return "#44BB44"  # Green


def enrich_chemicals(vlm_chemicals: list[dict]) -> list[dict]:
    """Enrich VLM-extracted chemicals with reference DB data."""
    enriched = []
    for chem in vlm_chemicals:
        name = chem.get("name", "").lower().strip()
        # Lookup in DB (try various name formats)
        db_entry = CHEMICAL_DB.get(name)
        if not db_entry:
            # Try without underscores/spaces
            normalized = name.replace(" ", "_").replace("-", "_")
            db_entry = CHEMICAL_DB.get(normalized)
        if not db_entry:
            # Try partial match
            for key, val in CHEMICAL_DB.items():
                if key in name or name in key:
                    db_entry = val
                    break

        enriched.append({
            "name": chem.get("name", "Unknown"),
            "quantity": chem.get("quantity"),
            "category": db_entry["category"] if db_entry else chem.get("category", "other"),
            "risk_level": db_entry["risk_level"] if db_entry else chem.get("risk_level", "unknown"),
        })
    return enriched


def parse_quantity(qty_str) -> float:
    """Extract numeric value from quantity string like '5mg', '2.5ml', '100mcg'.
    
    Returns the numeric value for chart bar widths, or 1.0 as fallback.
    """
    if not qty_str:
        return 1.0
    match = re.search(r'([\d.]+)', str(qty_str))
    return float(match.group(1)) if match else 1.0


# --- Python Fallback Calculations ---

def calculate_theoretical_growth(
    ingredients: list[str],
    preservatives: list[str],
    shelf_life_days: int,
    days_since_mfg: int,
    spoilage_level: int,
) -> dict:
    """Calculate theoretical bacteria growth using logistic model.
    
    P(t) = K / (1 + ((K - P0) / P0) * e^(-rt))
    """
    P0 = 10  # Initial CFU
    K = 10000  # Carrying capacity
    r = 0.1  # Base growth rate

    # Sugars increase growth rate
    sugar_keywords = ["sucrose", "sugar", "sorbitol", "mannitol", "maltodextrin", "glucose", "fructose"]
    for ing in ingredients:
        if any(s in ing.lower() for s in sugar_keywords):
            r *= 1.3
            break

    # Preservatives decrease growth rate
    if preservatives:
        r *= 0.3
    else:
        r *= 1.5

    # High spoilage means contamination already present
    if spoilage_level > 50:
        P0 = 1000

    # Generate growth curve
    growth_curve = {}
    for day in [0, 30, 60, 90, 180, 365]:
        growth = K / (1 + ((K - P0) / P0) * math.exp(-r * day))
        growth_pct = min(100, int((growth / K) * 100))
        growth_curve[f"day_{day}"] = growth_pct

    # Find critical threshold day (when growth > 60%)
    critical_day = shelf_life_days
    for day in range(0, shelf_life_days + 1, 10):
        growth = K / (1 + ((K - P0) / P0) * math.exp(-r * day))
        if growth / K > 0.6:
            critical_day = day
            break

    current_day = min(days_since_mfg, 365)
    current_growth = K / (1 + ((K - P0) / P0) * math.exp(-r * current_day))
    current_growth_pct = min(100, int((current_growth / K) * 100))

    return {
        "current_growth_level": current_growth_pct,
        "growth_curve": growth_curve,
        "shelf_life_days": shelf_life_days,
        "critical_threshold_day": critical_day,
        "days_to_critical": max(0, critical_day - days_since_mfg),
        "factors": {
            "preservative_effectiveness": "high" if preservatives else "low",
            "storage_impact": "optimal",
            "visual_contamination": "severe" if spoilage_level > 60 else "mild" if spoilage_level > 30 else "none",
            "preservatives_detected": preservatives,
            "days_since_manufacturing": days_since_mfg,
        },
    }


def calculate_dynamic_expiry(
    mfg_date, exp_date, spoilage_assessment: dict,
    color_deviation: float, preservatives: list,
) -> dict:
    """Calculate dynamic expiry based on visual indicators.
    
    When dates are missing, defaults to a 1-year shelf life:
    - Missing mfg_date: assumes 180 days old
    - Missing exp_date: assumes 180 days remaining
    """
    today = datetime.now()

    if not mfg_date:
        mfg_date = today - timedelta(days=180)
    if not exp_date:
        exp_date = today + timedelta(days=180)

    shelf_life = (exp_date - mfg_date).days
    adjustment = 0
    factors = {}

    # Visual degradation
    if spoilage_assessment.get("discoloration"):
        reduction = int(shelf_life * 0.20)
        adjustment += reduction
        factors["visual_degradation"] = {"percentage": 20, "days_reduced": reduction, "reason": "discoloration"}

    if spoilage_assessment.get("cloudiness"):
        reduction = int(shelf_life * 0.10)
        adjustment += reduction
        factors["cloudiness"] = {"percentage": 10, "days_reduced": reduction, "reason": "cloudiness"}

    if spoilage_assessment.get("sediment"):
        reduction = int(shelf_life * 0.25)
        adjustment += reduction
        factors["sediment"] = {"percentage": 25, "days_reduced": reduction, "reason": "sediment"}

    if not spoilage_assessment.get("seal_intact", True):
        reduction = int(shelf_life * 0.30)
        adjustment += reduction
        factors["seal_damage"] = {"percentage": 30, "days_reduced": reduction, "reason": "broken seal"}

    # Color deviation
    if color_deviation > 0.3:
        reduction = int(shelf_life * color_deviation * 0.5)
        adjustment += reduction
        factors["color_deviation"] = {
            "percentage": int(color_deviation * 50),
            "days_reduced": reduction,
            "reason": f"color deviation {color_deviation:.2f}",
        }

    # No preservatives
    if not preservatives:
        reduction = int(shelf_life * 0.30)
        adjustment += reduction
        factors["no_preservatives"] = {"percentage": 30, "days_reduced": reduction, "reason": "no preservatives detected"}

    adjusted_shelf_life = max(30, shelf_life - adjustment)
    dynamic_expiry = mfg_date + timedelta(days=adjusted_shelf_life)
    today = datetime.now()

    return {
        "shelf_life_days": shelf_life,
        "adjusted_shelf_life_days": adjusted_shelf_life,
        "days_until_dynamic_expiry": max(0, (dynamic_expiry - today).days),
        "days_until_static_expiry": max(0, (exp_date - today).days),
        "adjustment_factors": factors,
        "total_adjustment_percentage": min(90, int((adjustment / shelf_life) * 100)),
        "dynamic_expiry_date": dynamic_expiry.strftime("%Y-%m-%d"),
        "static_expiry_date": exp_date.strftime("%Y-%m-%d"),
    }


def estimate_color_from_spoilage(spoilage_assessment: dict) -> dict:
    """Estimate color deviation from spoilage assessment."""
    deviation = 0.0
    indicators = []

    if spoilage_assessment.get("discoloration"):
        deviation += 0.4
        indicators.append({"type": "oxidation", "severity": "moderate", "confidence": 0.7})

    if spoilage_assessment.get("cloudiness"):
        deviation += 0.2
        indicators.append({"type": "turbidity", "severity": "mild", "confidence": 0.6})

    spoilage_level = spoilage_assessment.get("spoilage_level", 0)
    if spoilage_level > 60:
        deviation += 0.3
        indicators.append({"type": "general_degradation", "severity": "severe", "confidence": 0.8})
    elif spoilage_level > 30:
        deviation += 0.1
        indicators.append({"type": "general_degradation", "severity": "mild", "confidence": 0.6})

    return {
        "color_deviation": min(1.0, deviation),
        "color_spoilage_score": int(min(1.0, deviation) * 100),
        "degradation_indicators": indicators,
        "estimated_days_since_optimal": int(deviation * 180),
    }
