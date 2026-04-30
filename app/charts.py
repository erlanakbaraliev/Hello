"""Plotly helpers: light-theme templates and layout."""

from __future__ import annotations

PLOT_FONT = "Source Sans 3, Segoe UI, system-ui, sans-serif"


def plotly_template() -> str:
    return "plotly_white"


def plotly_label_color() -> str:
    return "#334155"


def plotly_title_color() -> str:
    return "#0f172a"


def plotly_paper_bg() -> str:
    return "rgba(255,255,255,0.92)"


def plotly_plot_bg() -> str:
    return "#fafbfc"


def plotly_grid_color() -> str:
    return "#e2e8f0"


def plotly_legend_bg() -> str:
    return "rgba(255,255,255,0.85)"


def plotly_legend_border() -> str:
    return "#e2e8f0"
