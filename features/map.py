# features/map.py
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.graph_objs as go
from .utils import hex_to_pixel, polygon_hex, slug_name, PARTY_COLOR_MAP

# keep YOUR first so it gets priority where present
DEFAULT_PARTY_ORDER = ["YOUR", "LAB", "CON", "RFM", "LDM", "GRN", "MIN", "Oth", "SNP", "PLC"]


def _safe_float(val):
    """
    Convert val to float safely. Return 0.0 for '-', '', None or any non-convertible value.
    Handles numeric types and string forms.
    """
    try:
        if val is None:
            return 0.0
        # if it's already numeric, cast to float directly
        if isinstance(val, (int, float,)):
            return float(val)
        v = str(val).strip()
        if v == "" or v == "-":
            return 0.0
        # remove commas (just in case)
        v = v.replace(",", "")
        return float(v)
    except Exception:
        return 0.0


def build_map_figure(data_dir: Path, nowcast: pd.DataFrame) -> Tuple[go.Figure, Dict[str, Tuple[float, float]]]:
    """
    Build a map figure using numeric party share columns from the provided nowcast DataFrame.

    Important behavior:
    - Party columns detected from DEFAULT_PARTY_ORDER are coerced to numeric here.
    - The "winner" for a hex is selected from numeric shares (max share).
      The original 'Winner' column is used only as a fallback when numeric shares are all zero/missing.
    """
    hexjson_path = data_dir / "uk-constituencies-2024.hexjson"
    with open(hexjson_path, "r", encoding="utf-8") as f:
        hexjson = json.load(f)

    # detect party columns in the incoming DataFrame (priority from DEFAULT_PARTY_ORDER)
    parties = [p for p in DEFAULT_PARTY_ORDER if p in nowcast.columns]
    if not parties:
        exclude = {"Constituency", "constituency", "ConstituencyName", "Constituency Name", "Current", "Winner", "__slug"}
        parties = [c for c in nowcast.columns if c not in exclude][:9]

    # Work on a local copy to avoid mutating caller's DataFrame
    df = nowcast.copy()

    # Ensure the party columns we care about are numeric (coerce strings -> floats, replace NaN with 0.0)
    numeric_party_cols = [p for p in parties if p in df.columns]
    if numeric_party_cols:
        df[numeric_party_cols] = df[numeric_party_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # index by slug for fast lookup; assume caller provided __slug
    if "__slug" not in df.columns:
        df["__slug"] = df.get("Constituency", "").apply(slug_name)
    nowcast_index = df.set_index("__slug", drop=False)

    hex_entries = hexjson.get("hexes", {})
    trace_polygons = []
    trace_centroids = []
    slug_to_centroid = {}

    for hid, meta in hex_entries.items():
        name = meta.get("n") or meta.get("name") or hid
        q = meta.get("q")
        r = meta.get("r")
        hex_colour = meta.get("colour")

        slug = slug_name(name)
        now_row = nowcast_index.loc[slug] if slug in nowcast_index.index else None

        # Determine winner using numeric shares first
        winner = None
        if now_row is not None:
            # collect numeric shares in the same order as numeric_party_cols
            vals = [_safe_float(now_row.get(p, 0.0)) for p in numeric_party_cols]
            if any(v > 0.0 for v in vals):
                # choose index of the maximum value safely (no pandas.np)
                max_ix = max(range(len(vals)), key=lambda i: vals[i])
                winner = numeric_party_cols[max_ix]
            else:
                # fallback to the original Winner column if present and informative
                w = now_row.get("Winner", None) if "Winner" in now_row else None
                if w is not None and str(w).strip() not in ("", "-", "nan"):
                    winner = w
                else:
                    winner = None

        # pick colour from computed winner; fallback to hexjson colour or neutral grey
        color = PARTY_COLOR_MAP.get(str(winner), hex_colour or "#CCCCCC")

        cx, cy = hex_to_pixel(q, r)
        verts = polygon_hex(cx, cy)
        xs, ys = zip(*verts)

        # polygon trace for the hex
        poly = go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            mode="lines",
            line=dict(width=0.6, color="white"),
            fillcolor=color,
            hoverinfo="none",
            name=name,
            showlegend=False,
        )
        trace_polygons.append(poly)

        # centroid invisible marker with hover text = name
        centroid = go.Scatter(
            x=[cx],
            y=[cy],
            mode="markers",
            marker=dict(size=44, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=[name],
            hovertemplate="%{text}<extra></extra>",
            name=f"{name}-hit",
            showlegend=False,
        )
        trace_centroids.append(centroid)
        slug_to_centroid[slug] = (cx, cy)

    all_traces = trace_polygons + trace_centroids
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x"),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # tighten axis ranges
    all_x = [x for tr in all_traces for x in tr.x if x is not None]
    all_y = [y for tr in all_traces for y in tr.y if y is not None]
    if all_x and all_y:
        minx, maxx = min(all_x), max(all_x)
        miny, maxy = min(all_y), max(all_y)
        dx = (maxx - minx) * 0.005
        dy = (maxy - miny) * 0.005
        fig.update_xaxes(range=[minx - dx, maxx + dx])
        fig.update_yaxes(range=[miny - dy, maxy + dy], autorange=False)

    return fig, slug_to_centroid


def find_centroid_by_slug(slug_to_centroid: Dict[str, Tuple[float, float]], slug: str):
    return slug_to_centroid.get(slug)
