import plotly.graph_objects as go
from typing import Dict, Any


def plot_style_radar(stats_dict: Dict[str, Any]) -> go.Figure:
    """Create a simple radar chart for style metrics.

    Parameters
    ----------
    stats_dict: Dict[str, Any]
        Mapping of metric name to numeric value. ``None`` values are ignored.

    Returns
    -------
    go.Figure
        Plotly radar chart visualising the provided metrics.
    """
    if not stats_dict:
        return go.Figure()

    categories = list(stats_dict.keys())
    values = [float(stats_dict[c]) for c in categories]

    # Close the radar shape
    categories += categories[:1]
    values += values[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=values, theta=categories, fill="toself", name="")
    )

    max_val = max(values) if values else 1
    min_val = min(values) if values else 0

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[min(0, min_val), max_val * 1.1])),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
