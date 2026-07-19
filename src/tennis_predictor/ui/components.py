from __future__ import annotations

import html
from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def hero(title: str, subtitle: str, eyebrow: str = "TENNIS EDGE LAB") -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "") -> str:
    detail_html = f'<div class="metric-detail">{html.escape(detail)}</div>' if detail else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f"{detail_html}</div>"
    )


def render_metric_row(metrics: Iterable[tuple[str, str, str]]) -> None:
    items = list(metrics)
    columns = st.columns(len(items))
    for column, (label, value, detail) in zip(columns, items, strict=False):
        with column:
            st.markdown(metric_card(label, value, detail), unsafe_allow_html=True)


def probability_gauge(player1: str, probability: float) -> go.Figure:
    probability = float(np.clip(probability, 0.0, 1.0))
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": f"{player1} win probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#b7ff3c", "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,91,91,0.14)"},
                    {"range": [40, 60], "color": "rgba(255,255,255,0.06)"},
                    {"range": [60, 100], "color": "rgba(183,255,60,0.10)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=35, r=35, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f5f7fa"},
    )
    return figure


def probability_matchup(player1: str, player2: str, p1: float) -> None:
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = 1.0 - p1
    st.markdown(
        f"""
        <div class="matchup-shell">
          <div class="player-side player-one">
            <div class="player-name">{html.escape(player1)}</div>
            <div class="player-prob">{p1:.1%}</div>
          </div>
          <div class="versus">VS</div>
          <div class="player-side player-two">
            <div class="player-name">{html.escape(player2)}</div>
            <div class="player-prob">{p2:.1%}</div>
          </div>
          <div class="probability-track">
            <div class="probability-fill" style="width:{p1 * 100:.2f}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, active: bool, unknown: bool = False) -> str:
    css_class = "unknown" if unknown else ("active" if active else "clear")
    value = "UNKNOWN" if unknown else ("FLAGGED" if active else "CLEAR")
    return (
        f'<span class="status-badge {css_class}">'
        f'{html.escape(label)} · {value}</span>'
    )


def format_percent(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"
