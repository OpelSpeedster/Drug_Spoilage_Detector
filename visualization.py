"""Plotly chart generators for spoilage visualization.

Four charts:
1. Chemical composition bar chart
2. Spoilage timeline (Gantt-style)
3. Bacteria growth gauge
4. Risk radar chart
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils import parse_quantity


RISK_COLORS = {
    "safe": "#44BB44",
    "caution": "#FFAA00",
    "danger": "#FF4444",
    "unknown": "#888888",
}

CATEGORY_COLORS = {
    "active_ingredient": "#3366CC",
    "preservative": "#DC3912",
    "solvent": "#FF9900",
    "binder": "#109618",
    "filler": "#990099",
    "disintegrant": "#0099C6",
    "lubricant": "#DD4477",
    "colorant": "#66AA00",
    "sweetener": "#B82E2E",
    "flavoring": "#316395",
    "acidifier": "#994499",
    "buffer": "#22AA99",
    "ph_adjuster": "#AAAA11",
    "excipient": "#6633CC",
    "other": "#888888",
}


def _empty_fig(text: str, height: int = 250) -> go.Figure:
    """Return a dark-themed empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888888"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        height=height,
    )
    return fig


def create_chemical_bar_chart(chemicals: list[dict]) -> go.Figure:
    """Horizontal bar chart showing chemical name, quantity, and risk level."""
    if not chemicals:
        fig = go.Figure()
        fig.add_annotation(
            text="No chemicals detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888888"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=300,
        )
        return fig

    names = [c.get("name", "Unknown") for c in chemicals]
    quantities = [c.get("quantity") or "" for c in chemicals]
    categories = [c.get("category", "other") for c in chemicals]
    risk_levels = [c.get("risk_level", "unknown") for c in chemicals]
    colors = [RISK_COLORS.get(r, "#888888") for r in risk_levels]
    values = [parse_quantity(q) for q in quantities]

    labels = []
    for name, qty, cat, risk in zip(names, quantities, categories, risk_levels):
        qty_str = f" | {qty}" if qty else ""
        labels.append(f"{name}{qty_str} [{cat}]")

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="inside",
        textfont=dict(color="white", size=11),
    ))

    fig.update_layout(
        title=dict(text="Chemical Composition (Name → Quantity)", font=dict(color="white", size=16)),
        xaxis=dict(title="Quantity", tickfont=dict(color="white", size=10), gridcolor="#333333"),
        yaxis=dict(autorange="reversed", tickfont=dict(color="white", size=11)),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=max(300, len(names) * 40 + 100),
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )

    for risk, color in RISK_COLORS.items():
        if risk in risk_levels:
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color=color,
                name=risk.capitalize(),
                showlegend=True,
            ))

    return fig


def create_spoilage_timeline(
    mfg_date: datetime | None,
    exp_date: datetime | None,
    predicted_spoilage: datetime | None,
    today: datetime | None = None,
) -> go.Figure:
    """Gantt-style timeline with two modes:

    Mode 1 (exp_date known): Static expiry is primary — shows shelf life + remaining days.
    Mode 2 (no exp_date):    Visual estimate is primary — shows predicted days from spoilage score.
    """
    if today is None:
        today = datetime.now()

    fig = go.Figure()

    if exp_date:
        # Mode 1: Label has dates — show full shelf life timeline
        days_left = (exp_date - today).days
        expired = days_left < 0

        # Shelf life bar (MFG → Expiry)
        if mfg_date:
            fig.add_trace(go.Bar(
                x=[(exp_date - mfg_date).days], y=["Shelf Life"],
                orientation="h",
                base=mfg_date.strftime("%Y-%m-%d"),
                marker_color="#2244AA",
                text=f"{mfg_date.strftime('%b %Y')} → {exp_date.strftime('%b %Y')}",
                textposition="inside", textfont=dict(color="white"),
                showlegend=False,
            ))

        # Remaining / Overdue bar
        color = "#FF4444" if expired else "#44BB44"
        label = f"Expired {abs(days_left)}d ago" if expired else f"{days_left}d left"
        base = exp_date if expired else today

        fig.add_trace(go.Bar(
            x=[abs(days_left)], y=["Remaining"],
            orientation="h",
            base=base.strftime("%Y-%m-%d"),
            marker_color=color,
            text=label,
            textposition="inside", textfont=dict(color="white", size=12),
            showlegend=False,
        ))

    else:
        # Mode 2: No label dates — visual estimate only
        if predicted_spoilage:
            est_days = (predicted_spoilage - today).days
            color = "#FF4444" if est_days < 0 else "#FFAA00"
            fig.add_trace(go.Bar(
                x=[max(30, abs(est_days))], y=["Visual Estimate"],
                orientation="h",
                base=today.strftime("%Y-%m-%d"),
                marker_color=color,
                text=f"Visual only — ~{max(0, est_days)} days",
                textposition="inside", textfont=dict(color="white", size=12),
                showlegend=False,
            ))
        else:
            return _empty_fig("No date information available", height=200)

    # Today marker
    fig.add_vline(
        x=today.strftime("%Y-%m-%d"),
        line_dash="dash", line_color="white", line_width=2,
        annotation_text="Today", annotation_font=dict(color="white"),
    )

    # Expiry marker (Mode 1 only)
    if exp_date:
        exp_color = "#FF4444" if (exp_date - today).days < 0 else "#FFAA00"
        fig.add_vline(
            x=exp_date.strftime("%Y-%m-%d"),
            line_dash="dot", line_color=exp_color,
            annotation_text=f"Expiry {exp_date.strftime('%b %Y')}",
            annotation_font=dict(color="#FFAA00"),
        )

    fig.update_layout(
        title=dict(text="Expiry Timeline", font=dict(color="white", size=16)),
        xaxis=dict(type="date", tickfont=dict(color="white"), gridcolor="#333"),
        yaxis=dict(tickfont=dict(color="white", size=12)),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        height=200, barmode="overlay",
        margin=dict(l=110, r=20, t=50, b=30),
    )
    return fig


def create_bacteria_gauge(growth_level: int) -> go.Figure:
    """Gauge chart showing bacteria growth level (0-100)."""
    growth_level = max(0, min(100, growth_level))

    if growth_level > 60:
        bar_color = "#FF4444"
    elif growth_level > 30:
        bar_color = "#FFAA00"
    else:
        bar_color = "#44BB44"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=growth_level,
        number=dict(suffix="/100", font=dict(color="white", size=28)),
        title=dict(text="Bacteria Growth", font=dict(color="white", size=16)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="white")),
            bar=dict(color=bar_color),
            bgcolor="#16213e",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="#1a3a1a"),
                dict(range=[30, 60], color="#3a3a1a"),
                dict(range=[60, 100], color="#3a1a1a"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=growth_level,
            ),
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        height=250,
        margin=dict(l=20, r=20, t=60, b=10),
    )

    return fig


def create_risk_radar(
    visual_score: int,
    bacteria_score: int,
    date_score: int,
    chemical_risk: float,
) -> go.Figure:
    """Radar/spider chart showing multi-axis risk assessment."""
    categories = ["Visual", "Bacteria", "Date Proximity", "Chemical Risk"]
    values = [
        max(0, min(100, visual_score)),
        max(0, min(100, bacteria_score)),
        max(0, min(100, date_score)),
        max(0, min(100, int(chemical_risk * 100))),
    ]

    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(255, 68, 68, 0.2)",
        line=dict(color="#FF4444", width=2),
        marker=dict(size=8, color="#FF4444"),
    ))

    # Add safe zone reference
    safe_values = [30, 30, 30, 30] + [30]
    fig.add_trace(go.Scatterpolar(
        r=safe_values,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(68, 187, 68, 0.1)",
        line=dict(color="#44BB44", width=1, dash="dash"),
        marker=dict(size=0),
        name="Safe Zone",
    ))

    fig.update_layout(
        title=dict(text="Risk Assessment", font=dict(color="white", size=16)),
        polar=dict(
            bgcolor="#16213e",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="white", size=9),
                gridcolor="#333333",
            ),
            angularaxis=dict(
                tickfont=dict(color="white", size=11),
                gridcolor="#333333",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        height=350,
        margin=dict(l=40, r=40, t=60, b=20),
        showlegend=True,
        legend=dict(
            font=dict(color="white", size=10),
            x=0.8,
            y=-0.1,
        ),
    )

    return fig


def create_bacteria_growth_curve(
    growth_curve: dict,
    current_day: int,
    critical_threshold_day: int,
) -> go.Figure:
    """Line chart showing bacteria growth over time with critical threshold."""
    if not growth_curve or "growth_curve" not in growth_curve:
        fig = go.Figure()
        fig.add_annotation(
            text="No growth curve data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888888"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=300,
        )
        return fig

    curve_data = growth_curve["growth_curve"]
    days = sorted([int(k.split("_")[1]) for k in curve_data.keys()])
    values = [curve_data[f"day_{d}"] for d in days]

    fig = go.Figure()

    # Growth curve
    fig.add_trace(go.Scatter(
        x=days,
        y=values,
        mode="lines+markers",
        name="Bacteria Growth",
        line=dict(color="#FF6B6B", width=3),
        marker=dict(size=8, color="#FF6B6B"),
        fill="tozeroy",
        fillcolor="rgba(255, 107, 107, 0.2)",
    ))

    # Critical threshold line
    fig.add_hline(
        y=60,
        line_dash="dash",
        line_color="#FFAA00",
        annotation_text="Critical Threshold",
        annotation_font_color="#FFAA00",
    )

    # Current day marker
    fig.add_vline(
        x=current_day,
        line_dash="dot",
        line_color="#44BB44",
        annotation_text=f"Day {current_day}",
        annotation_font_color="#44BB44",
    )

    fig.update_layout(
        title=dict(text="Bacteria Growth Over Time", font=dict(color="white", size=16)),
        xaxis=dict(
            title="Days Since Manufacturing",
            tickfont=dict(color="white", size=10),
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="Growth Level (0-100)",
            tickfont=dict(color="white", size=10),
            gridcolor="#333333",
            range=[0, 100],
        ),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=350,
        margin=dict(l=50, r=20, t=60, b=50),
        showlegend=True,
        legend=dict(
            font=dict(color="white", size=10),
            x=0.7,
            y=0.95,
        ),
    )

    return fig


def create_color_degradation_timeline(
    color_analysis: dict,
    shelf_life_days: int,
) -> go.Figure:
    """Line chart showing color deviation over time."""
    if not color_analysis or "color_deviation" not in color_analysis:
        fig = go.Figure()
        fig.add_annotation(
            text="No color analysis data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888888"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=300,
        )
        return fig

    current_deviation = color_analysis["color_deviation"]
    estimated_days = color_analysis.get("estimated_days_since_optimal", 0)

    # Simulate degradation curve (exponential decay)
    import numpy as np
    days = np.linspace(0, shelf_life_days, 50)
    # Deviation increases over time, accelerating near end
    deviations = current_deviation * (days / max(estimated_days, 1)) ** 1.5
    deviations = np.clip(deviations, 0, 1)

    fig = go.Figure()

    # Color deviation curve
    fig.add_trace(go.Scatter(
        x=days,
        y=deviations,
        mode="lines",
        name="Color Deviation",
        line=dict(color="#FF9900", width=3),
        fill="tozeroy",
        fillcolor="rgba(255, 153, 0, 0.2)",
    ))

    # Threshold lines
    fig.add_hline(y=0.3, line_dash="dash", line_color="#44BB44", annotation_text="Minor")
    fig.add_hline(y=0.6, line_dash="dash", line_color="#FFAA00", annotation_text="Moderate")
    fig.add_hline(y=0.8, line_dash="dash", line_color="#FF4444", annotation_text="Severe")

    fig.update_layout(
        title=dict(text="Color Degradation Timeline", font=dict(color="white", size=16)),
        xaxis=dict(
            title="Days Since Manufacturing",
            tickfont=dict(color="white", size=10),
            gridcolor="#333333",
        ),
        yaxis=dict(
            title="Color Deviation (0-1)",
            tickfont=dict(color="white", size=10),
            gridcolor="#333333",
            range=[0, 1],
        ),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=300,
        margin=dict(l=50, r=20, t=60, b=50),
    )

    return fig


def create_dynamic_expiry_comparison(
    static_expiry_days: int,
    dynamic_expiry_days: int,
    adjustment_factors: dict,
) -> go.Figure:
    """Bar chart comparing static vs dynamic expiry with adjustment factors."""
    if static_expiry_days is None or dynamic_expiry_days is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for expiry comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#888888"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            height=300,
        )
        return fig

    fig = go.Figure()

    # Main comparison bars
    fig.add_trace(go.Bar(
        x=["Static Expiry", "Dynamic Expiry"],
        y=[static_expiry_days, dynamic_expiry_days],
        marker_color=["#3366CC", "#FF6B6B"],
        text=[f"{static_expiry_days} days", f"{dynamic_expiry_days} days"],
        textposition="inside",
        textfont=dict(color="white", size=14),
        name="Shelf Life",
    ))

    # Add adjustment factors as annotations
    if adjustment_factors:
        total_reduction = sum(
            f.get("days_reduced", 0) for f in adjustment_factors.values()
        )
        fig.add_annotation(
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            text=f"Total Reduction: {total_reduction} days",
            showarrow=False,
            font=dict(color="#FFAA00", size=12),
        )

    fig.update_layout(
        title=dict(text="Static vs Dynamic Expiry", font=dict(color="white", size=16)),
        xaxis=dict(tickfont=dict(color="white", size=12)),
        yaxis=dict(
            title="Days Until Expiry",
            tickfont=dict(color="white", size=10),
            gridcolor="#333333",
        ),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        height=300,
        margin=dict(l=50, r=20, t=60, b=80),
        showlegend=False,
    )

    return fig
