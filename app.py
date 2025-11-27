from pathlib import Path
import re

import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events

from features.map import build_map_figure, find_centroid_by_slug
from features.utils import slug_name

import math
import numpy as np

# Main: only show map (no right-hand panel)
st.title("UK Nowcast Model (General Elections)")
st.subheader("Simulating optimal Your Party strategies to avoid left vote splitting.")

# Configuration
DATA_DIR = Path("data")
HEXJSON_PATH = DATA_DIR / "uk-constituencies-2024.hexjson"
NOWCAST_PATH = DATA_DIR / "nowcast.csv"

st.set_page_config(layout="wide", page_title="UK Nowcast Model (General Elections)", initial_sidebar_state="expanded")

# Load and validate data
if not HEXJSON_PATH.exists():
    st.error(f"Missing hexjson file at {HEXJSON_PATH}. Place `uk-constituencies-2024.hexjson` in data/.")
    st.stop()

if not NOWCAST_PATH.exists():
    st.error(f"Missing nowcast CSV at {NOWCAST_PATH}. Place `nowcast.csv` in data/.")
    st.stop()

# read CSV
nowcast_raw = pd.read_csv(NOWCAST_PATH, dtype=str)  # read as str to sanitize reliably

# sanitize electorate -> int; turnout -> fraction; party columns -> numeric shares (0..1)
def parse_int_like(s):
    if pd.isna(s):
        return 0
    # remove commas, spaces, non-digits
    s = re.sub(r"[^\d]", "", str(s))
    return int(s) if s != "" else 0

def parse_pct_like(s):
    if pd.isna(s):
        return 0.0
    s = str(s).strip()
    # handle values like '49.30%' or '0.493' or '-'
    if s == "-" or s == "":
        return 0.0
    if s.endswith("%"):
        try:
            return float(s.strip("%")) / 100.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

# Standard party list (ensure 'YOUR' is included)
COMMON_PARTIES = ["YOUR", "LAB", "CON", "RFM", "LDM", "GRN", "MIN", "Oth", "SNP", "PLC"]

# Convert columns
# Find the constituency column name
constituency_col = None
for candidate in ["Constituency", "constituency", "ConstituencyName", "Constituency Name"]:
    if candidate in nowcast_raw.columns:
        constituency_col = candidate
        break
if constituency_col is None:
    constituency_col = nowcast_raw.columns[0]

# Parse electorate and turnout into numeric columns
nowcast = nowcast_raw.copy()
nowcast["Electorate_parsed"] = nowcast.get("Electorate", "").apply(parse_int_like)
nowcast["Turnout_parsed"] = nowcast.get("Turnout", "").apply(parse_pct_like)

# Parse / coerce party share columns to numeric (0..1). Some CSVs use '-' for missing.
# We'll make sure all COMMON_PARTIES exist as numeric columns; missing ones are created as 0.0
for p in COMMON_PARTIES:
    if p in nowcast.columns:
        nowcast[p] = pd.to_numeric(nowcast[p].replace("-", pd.NA), errors="coerce").fillna(0.0)
    else:
        nowcast[p] = 0.0

# create slug column for lookups
nowcast["__slug"] = nowcast[constituency_col].apply(slug_name)
nowcast_index = nowcast.set_index("__slug", drop=False)

# --- Simulation control in sidebar ---
st.sidebar.header("Simulation controls")
add_nonvoter_pct = st.sidebar.number_input(
    "% non-voters convinced to vote for Your Party",
    min_value=0,
    max_value=50,
    value=0,
    step=1,
    help="This converts a percentage of the non-voting electorate into votes for Your Party."
)

# NEW: % left voters converted to Your Party
add_left_pct = st.sidebar.number_input(
    "% left voters converted to Your Party",
    min_value=0,
    max_value=50,
    value=0,
    step=1,
    help="This takes a percentage of current left-party voters (LAB, LDM, GRN, SNP, PLC, MIN) and transfers them to Your Party. The taken voters are split across left parties proportional to their current left vote shares."
)

# --- Helper functions for seat counts and rendering row (unchanged visual style) ---
try:
    from features.utils import PARTY_COLOR_MAP
except Exception:
    PARTY_COLOR_MAP = {
        "YOUR": "#800080",
        "LAB": "#DC241f",
        "CON": "#0087DC",
        "SNP": "#FFF100",
        "LDM": "#FDBB30",
        "GRN": "#6AB023",
        "RFM": "#3BB7D5",
        "PLC": "#6E86FF",
        "MIN": "#A0A0A0",
        "Oth": "#999999",
    }

def compute_plurality_winners(df):
    """
    Compute plurality winner from numeric party share columns in df.
    This intentionally ignores any pre-existing 'Winner' column so results
    reflect simulated adjustments (e.g. YOUR votes).
    """
    # detect party columns (include COMMON_PARTIES if present)
    party_cols = [c for c in COMMON_PARTIES if c in df.columns]
    if not party_cols:
        exclude = {constituency_col, "Current", "Winner", "__slug", "Electorate", "Turnout", "Electorate_parsed", "Turnout_parsed"}
        party_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not party_cols:
        # nothing to compute
        return pd.Series("", index=df.index, dtype=str)

    # coerce to numeric and fillna with 0.0
    numeric = df[party_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # idxmax picks the column with largest share per row
    winners = numeric.idxmax(axis=1).fillna("").astype(str)
    # Rows where all zeros will pick the first column; make those empty instead
    row_max = numeric.max(axis=1).fillna(0.0)
    winners = winners.where(row_max > 0, other="")
    winners = winners.replace({"nan": ""})
    return winners


def render_seat_count_row(df, max_items_in_row=None):
    winners = compute_plurality_winners(df)
    counts = winners[winners != ""].value_counts().sort_values(ascending=False)
    parties = list(counts.index)
    if max_items_in_row is not None and len(parties) > max_items_in_row:
        top = parties[: max_items_in_row - 1]
        rest = parties[max_items_in_row - 1 :]
        top_counts = counts.loc[top].to_dict()
        other_count = int(counts.loc[rest].sum())
        display = top[:] + ["Other"]
        counts_map = {**{k:int(v) for k,v in top_counts.items()}, "Other": other_count}
    else:
        display = parties
        counts_map = {k:int(v) for k,v in counts.astype(int).to_dict().items()}

    if not display:
        st.info("No plurality winners available to compute seat counts.")
        return

    cols = st.columns(len(display), gap="small")
    for col, party in zip(cols, display):
        with col:
            party_count = counts_map.get(party, 0)
            color = PARTY_COLOR_MAP.get(party, "#888888") if party != "Other" else "#999999"
            col.markdown(
                f"""
                <div style="background:#ffffff;border-radius:8px;padding:10px;
                            border:1px solid rgba(0,0,0,0.04);text-align:center;">
                  <div style="font-size:12px;color:#666;margin-bottom:6px;">
                      <span style="display:inline-block;width:12px;height:12px;background:{color};
                                   border-radius:3px;margin-right:8px;vertical-align:middle;
                                   border:1px solid rgba(0,0,0,0.06);"></span>
                      <span style="vertical-align:middle;font-weight:600;">{party}</span>
                  </div>
                  <div style="font-size:26px;font-weight:700;color:{color};">
                      {party_count}
                  </div>
                  <div style="font-size:11px;color:#888;margin-top:6px;">
                      projected seats
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- Build adjusted dataframe according to slider ---
def apply_nonvoter_conversion(df, pct_nonvoter, pct_left=0):
    """
    Defensive implementation that freezes original shares from the passed-in DataFrame
    and computes absolute votes for other parties from those frozen shares. Caller
    should pass the baseline (original) DataFrame to guarantee correct behaviour.

    New behaviour: pct_left is the percent of left-party voters (LAB, LDM, GRN, SNP, PLC, MIN)
    that are converted to 'YOUR'. The voters taken are removed proportionally from
    each left party's absolute votes and added to YOUR. This is applied per-constituency.
    """
    # work on a copy so we never mutate caller's DataFrame
    base = df.copy(deep=True)
    input_frac = float(pct_nonvoter) / 100.0
    left_frac = float(pct_left) / 100.0

    # party columns present (in baseline)
    party_cols = [p for p in COMMON_PARTIES if p in base.columns]

    # ensure numeric types for baseline shares and electorate/turnout
    for p in party_cols:
        base[p] = pd.to_numeric(base[p], errors="coerce").fillna(0.0)
    base["_electorate"] = pd.to_numeric(base["Electorate_parsed"], errors="coerce").fillna(0).astype(int)
    base["_turnout"] = pd.to_numeric(base["Turnout_parsed"], errors="coerce").fillna(0.0)

    # 1) current votes and non-voters (integers)
    base["_current_votes"] = (base["_electorate"] * base["_turnout"]).round().astype(int)
    base["_non_voters"] = (base["_electorate"] - base["_current_votes"]).clip(lower=0).astype(int)

    # 2) freeze baseline shares explicitly (safe even if df was adjusted earlier)
    for p in party_cols:
        base[f"_baseline_share_{p}"] = base[p].astype(float)

    # 3) compute absolute votes for other parties from frozen baseline shares
    for p in party_cols:
        base[f"_votes_{p}"] = (base["_current_votes"] * base[f"_baseline_share_{p}"]).round().astype(int)

    # 4) desired and actual new YOUR votes (from non-voter pool)
    base["_desired_your"] = (base["_electorate"] * (1.0 - base["_turnout"]) * input_frac).round().astype(int)
    base["_new_your_votes"] = base[["_desired_your", "_non_voters"]].min(axis=1).astype(int)

    # ensure YOUR vote buckets exist and add new votes from non-voters
    if "YOUR" not in base.columns:
        base["YOUR"] = 0.0
    if "_votes_YOUR" not in base.columns:
        base["_votes_YOUR"] = 0
    base["_votes_YOUR"] = base["_votes_YOUR"].astype(int) + base["_new_your_votes"].astype(int)

    # 5) LEFT-VOTER TRANSFER: take left_frac of each left party's absolute votes and transfer to YOUR
    left_parties = ["LAB", "LDM", "GRN", "SNP", "PLC", "MIN"]
    # only consider left parties that actually exist in this dataset
    left_parties_present = [p for p in left_parties if p in base.columns and f"_votes_{p}" in base.columns]

    # compute per-party taken voters and subtract from left parties' absolute votes
    # sum all taken voters to _new_from_left and add to YOUR
    if left_frac > 0 and left_parties_present:
        # compute taken voters per left party
        taken_cols = []
        for p in left_parties_present:
            taken_col = f"_left_take_{p}"
            # compute as rounded int of the absolute votes * left_frac
            base[taken_col] = (base[f"_votes_{p}"] * left_frac).round().astype(int)
            # subtract taken voters from the left-party vote absolute counts
            base[f"_votes_{p}"] = (base[f"_votes_{p}"] - base[taken_col]).clip(lower=0).astype(int)
            taken_cols.append(taken_col)

        # total new votes taken from left parties
        base["_new_from_left"] = base[taken_cols].sum(axis=1).astype(int)

        # add those to YOUR absolute votes
        base["_votes_YOUR"] = base["_votes_YOUR"].astype(int) + base["_new_from_left"].astype(int)
    else:
        # no left transfer
        base["_new_from_left"] = 0

    # 6) new total votes and safe denominator
    # Note: transfers from left do not change total voter count; only non-voter conversion increases totals.
    base["_new_total_votes"] = base["_current_votes"] + base["_new_your_votes"]
    base["_new_total_votes_safe"] = base["_new_total_votes"].replace({0: 1})

    # 7) recompute shares from absolute votes (other parties keep absolute votes, left parties were reduced above)
    for p in party_cols:
        base[p] = (base[f"_votes_{p}"] / base["_new_total_votes_safe"]).fillna(0.0)

    # YOUR share from its absolute votes
    base["YOUR"] = (base["_votes_YOUR"] / base["_new_total_votes_safe"]).fillna(0.0)

    # 8) return only expected downstream columns (plus internal debug cols if you need them)
    keep_cols = [constituency_col, "__slug", "Electorate_parsed", "Turnout_parsed"]
    keep_cols += [c for c in COMMON_PARTIES if c in base.columns]
    if "Winner" in base.columns:
        keep_cols.append("Winner")

    # include internal debug columns (absolute votes and intermediates)
    debug_cols = [c for c in base.columns if c.startswith("_votes_")]
    debug_cols += [c for c in ("_current_votes", "_non_voters", "_desired_your", "_new_your_votes", "_votes_YOUR", "_new_total_votes", "_new_from_left")]
    # include left_take columns if present
    debug_cols += [c for c in base.columns if c.startswith("_left_take_")]
    # keep only those that actually exist
    debug_cols = [c for c in debug_cols if c in base.columns]

    keep_cols += debug_cols

    return base[keep_cols].copy()

# Apply simulation
baseline_nowcast = nowcast.copy(deep=True)
adjusted = apply_nonvoter_conversion(baseline_nowcast, add_nonvoter_pct, add_left_pct)

# Render seat row from adjusted data
render_seat_count_row(adjusted, max_items_in_row=8)

# ---- projected seats bar plot ----
winners = compute_plurality_winners(adjusted)

# tally and ensure YOUR is shown even if zero
counts = winners[winners != ""].value_counts().astype(int)
if "YOUR" in adjusted.columns and "YOUR" not in counts.index:
    counts = pd.concat([pd.Series({"YOUR": 0}), counts]).sort_values(ascending=False)

# sort descending (largest first)
counts = counts.sort_values(ascending=False)

if counts.empty:
    st.info("No projected seats to display.")
else:
    parties = counts.index.tolist()
    values = counts.values.tolist()
    colors = [PARTY_COLOR_MAP.get(p, "#999999") for p in parties]

    bar_fig = go.Figure(
        go.Bar(
            x=values,
            y=parties,
            orientation="h",
            marker=dict(color=colors),
            text=values,
            textposition="auto"
        )
    )
    bar_fig.update_layout(
        title="",
        xaxis_title="Seats",
        yaxis_title="Party",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    bar_fig.update_traces(textangle=0)

    st.plotly_chart(
        bar_fig,
        width='stretch',
        config={"displayModeBar": False}
    )
# ---- end bar plot ----

# Sidebar: searchable selectbox (uses original nowcast names)
st.sidebar.header("Map controls")
constituency_options = [""] + sorted(nowcast[constituency_col].astype(str).tolist())
selected_constituency = st.sidebar.selectbox(
    "Find constituency",
    options=constituency_options,
    index=0,
    help="Type to filter the list, then select a constituency."
)

# Build figure (centroid markers include hover text = constituency name) using adjusted df
fig, slug_to_centroid = build_map_figure(DATA_DIR, adjusted)

# render the map via plotly_events so we can capture clicks
FIG_HEIGHT = 900
events = plotly_events(fig, click_event=True, hover_event=False, override_height=FIG_HEIGHT)

# utility: find slug from clicked coordinates (tolerance to avoid float mismatches)
def find_slug_from_click(slug_map, x_clicked, y_clicked, tol=1e-6):
    for slug, (cx, cy) in slug_map.items():
        if math.isclose(cx, x_clicked, rel_tol=0.0, abs_tol=tol) and math.isclose(cy, y_clicked, rel_tol=0.0, abs_tol=tol):
            return slug
    return None

# Determine selected slug either from click or sidebar selectbox
selected_slug = None

# prefer last click event if available
if events:
    # find last plotly_click
    for ev in reversed(events):
        if isinstance(ev, dict) and ev.get("event") == "plotly_click":
            pts = ev.get("points") or []
            if pts:
                pt = pts[0]
                x_clicked = pt.get("x")
                y_clicked = pt.get("y")
                # try exact match first, otherwise try small tolerance search
                selected_slug = find_slug_from_click(slug_to_centroid, x_clicked, y_clicked, tol=1e-6)
                if selected_slug is None:
                    selected_slug = find_slug_from_click(slug_to_centroid, x_clicked, y_clicked, tol=1e-3)
            break

# fallback to sidebar selection if no click
if selected_slug is None and selected_constituency:
    selected_slug = slug_name(selected_constituency)

# show debug panel as an expander below the map
with st.expander("Constituency details"):
    if not selected_slug:
        st.write("No constituency selected. Click a hex on the map or choose one from the sidebar.")
    else:
        if selected_slug not in adjusted.set_index("__slug").index:
            st.write(f"Selected slug '{selected_slug}' not found in adjusted data.")
        else:
            # fetch row for slug; normalize to a single Series
            sel = adjusted.set_index("__slug").loc[selected_slug]
            if isinstance(sel, pd.DataFrame):
                # deterministic: take the first matching row
                row = sel.iloc[0]
            else:
                row = sel  # already a Series

            # parse inputs (must be done after row is selected)
            electorate = int(row.get("Electorate_parsed", 0))
            turnout = float(row.get("Turnout_parsed", 0.0))
            current_votes = int(round(electorate * turnout))
            non_voters = electorate - current_votes
            input_frac = float(add_nonvoter_pct) / 100.0
            desired_your = int(round(electorate * (1.0 - turnout) * input_frac))
            new_your_votes = min(desired_your, non_voters)

            new_total_votes = current_votes + new_your_votes
            new_total_votes_safe = new_total_votes if new_total_votes > 0 else 1

            # compute existing absolute votes for each party using authoritative _votes_ columns when present
            vote_rows = {}
            for p in [c for c in COMMON_PARTIES if c in adjusted.columns]:
                votes_col = f"_votes_{p}"
                if votes_col in row.index:
                    val = row[votes_col]
                    # safe conversion to int for various possible types
                    try:
                        votes_abs = int(val)
                    except Exception:
                        # try .item() for numpy/pandas scalars, else handle array-like
                        try:
                            votes_abs = int(val.item())
                        except Exception:
                            if hasattr(val, "__len__") and len(val) > 0:
                                votes_abs = int(val[0])
                            else:
                                votes_abs = int(float(val))
                else:
                    # fallback: infer from current_votes * original_share (only used if _votes_ not present)
                    votes_abs = int(round(current_votes * (row.get(p, 0.0) if p != "YOUR" else 0.0)))
                vote_rows[p] = votes_abs

            # Prefer any explicit internal fields for exactness
            if "_new_your_votes" in row.index:
                new_your_votes = int(row["_new_your_votes"])
            if "_current_votes" in row.index:
                current_votes = int(row["_current_votes"])
                # recompute dependent totals just in case
                non_voters = electorate - current_votes
                new_total_votes = current_votes + new_your_votes
                new_total_votes_safe = new_total_votes if new_total_votes > 0 else 1

            # recompute final shares from authoritative absolute votes
            # small helper: coerce a scalar/array-like into an int safely
            def _safe_int_from_val(val, default=0):
                # handle None / NaN
                try:
                    if val is None:
                        return int(default)
                except Exception:
                    pass
                # pandas NA / np.nan
                try:
                    if pd.isna(val):
                        return int(default)
                except Exception:
                    pass
                # Try straightforward conversion first
                try:
                    return int(val)
                except Exception:
                    pass
                # Try .item() (numpy / pandas scalar)
                try:
                    return int(val.item())
                except Exception:
                    pass
                # If array-like/Series, take the first element
                try:
                    if hasattr(val, "__len__") and len(val) > 0:
                        return int(val[0])
                except Exception:
                    pass
                # Last resort: float conversion
                try:
                    return int(float(val))
                except Exception:
                    return int(default)

            # recompute final shares from authoritative absolute votes
            final_shares = {}
            for p, v in vote_rows.items():
                if p == "YOUR":
                    v_final = _safe_int_from_val(row.get("_votes_YOUR", new_your_votes), default=new_your_votes)
                else:
                    v_final = int(v)  # v comes from vote_rows and is already int
                final_shares[p] = v_final / new_total_votes_safe

            # prepare display table
            table_rows = []
            for p in sorted(final_shares.keys(), key=lambda x: -final_shares[x]):
                votes_col = f"_votes_{p}"
                # Prefer authoritative _votes_<party> column when present, but coerce safely
                if votes_col in row.index:
                    votes_abs = _safe_int_from_val(row.loc[votes_col], default=vote_rows.get(p, 0))
                else:
                    votes_abs = int(vote_rows.get(p, 0))

                # Ensure YOUR uses the explicit _votes_YOUR field when available
                if p == "YOUR":
                    votes_abs = _safe_int_from_val(row.get("_votes_YOUR", new_your_votes), default=new_your_votes)

                pct = final_shares[p]
                table_rows.append((p, votes_abs, f"{pct:.1%}"))

            df_dbg = pd.DataFrame(table_rows, columns=["Party", "Absolute votes", "Percent total votes"])
            st.markdown(f"**Constituency:** {row.get(constituency_col, selected_slug)}")
            st.write(f"Electorate: {electorate:,}")
            st.write(f"Turnout: {turnout + input_frac:.2%}  →  Voters: {new_total_votes:,}")
            st.write(f"Non-voters: {electorate - new_total_votes:,}")
            st.write(f"Total Your Party votes: {desired_your:,}")
            st.dataframe(df_dbg, width='content')

            # --------------------------
            # New: grid plot of winners for combinations of non-voter % (x) and left-voter % (y)
            # We'll compute a grid from 0..50 (inclusive) for both axes with step 1.
            # For each cell we simulate just this constituency using apply_nonvoter_conversion
            # on a one-row baseline and read the plurality winner.
            # --------------------------

            # Prepare baseline one-row df for simulation (use baseline_nowcast original shares)
            # Ensure we produce a DataFrame with a __slug column (not just an index)
            if selected_slug not in baseline_nowcast.set_index("__slug").index:
                st.error("Selected constituency not found in baseline data.")

            # --------------------------
            # New: grid plot of winners for combinations of non-voter % (x) and left-voter % (y)
            # Compute a 0..50 x 0..50 grid and display a heatmap inside the expander.
            # --------------------------

            # Build a one-row baseline DataFrame for the selected constituency.
            baseline_row = baseline_nowcast[baseline_nowcast["__slug"] == selected_slug].copy()
            if baseline_row.shape[0] == 0:
                st.error("Selected constituency not found in baseline data.")
            else:
                # Ensure a proper DataFrame with columns (not index)
                baseline_row = baseline_row.reset_index(drop=True)

                # Ensure constituency_col and __slug exist as columns (apply_nonvoter_conversion expects them)
                if "__slug" not in baseline_row.columns:
                    baseline_row["__slug"] = selected_slug
                if constituency_col not in baseline_row.columns:
                    try:
                        baseline_row[constituency_col] = nowcast_index.loc[selected_slug, constituency_col]
                    except Exception:
                        baseline_row[constituency_col] = selected_slug

                # Ensure numeric columns are present so apply_nonvoter_conversion won't drop them later
                for col in ("Electorate_parsed", "Turnout_parsed"):
                    if col not in baseline_row.columns:
                        baseline_row[col] = nowcast_index.loc[selected_slug, col] if col in nowcast_index.columns else 0
                for p in COMMON_PARTIES:
                    if p not in baseline_row.columns:
                        baseline_row[p] = 0.0

                # Grid axes
                x_vals = list(range(0, 51))  # % non-voters (x-axis)
                y_vals = list(range(0, 51))  # % left voters (y-axis)

                # Parties present for this constituency
                parties_present = [p for p in COMMON_PARTIES if p in baseline_row.columns]
                party_codes = {p: i for i, p in enumerate(parties_present)}
                code_to_party = {i: p for p, i in party_codes.items()}
                colors_for_codes = [PARTY_COLOR_MAP.get(code_to_party[i], "#888888") for i in range(len(party_codes))]

                # Pre-allocate
                z = np.full((len(y_vals), len(x_vals)), fill_value=-1, dtype=int)
                hovertext = [["" for _ in x_vals] for _ in y_vals]

                # Compute grid winners
                for iy, left_pct in enumerate(y_vals):
                    for ix, nonv_pct in enumerate(x_vals):
                        sim = apply_nonvoter_conversion(baseline_row, pct_nonvoter=nonv_pct, pct_left=left_pct)
                        winner_series = compute_plurality_winners(sim)
                        winner = ""
                        if len(winner_series) > 0:
                            val = winner_series.iloc[0]
                            if pd.notna(val):
                                winner = str(val)

                        if winner == "" or winner not in party_codes:
                            # fallback: prefer Oth if present, else pick first party
                            if "Oth" in party_codes:
                                code = party_codes["Oth"]
                                winner_label = "Oth"
                            else:
                                code = 0
                                winner_label = code_to_party.get(code, "")
                        else:
                            code = party_codes[winner]
                            winner_label = winner

                        z[iy, ix] = int(code)
                        hovertext[iy][ix] = f"{winner_label}<br>% non-voters: {nonv_pct}%<br>% left converted: {left_pct}%"

                # Build colorscale
                n_codes = len(colors_for_codes)
                if n_codes == 0:
                    st.write("No party columns available to build winner grid.")
                else:
                    # Build discrete (flat) colorscale so bands are solid, not blended
                    if n_codes == 1:
                        colorscale = [[0.0, colors_for_codes[0]], [1.0, colors_for_codes[0]]]
                    else:
                        colorscale = []
                        # create contiguous same-color intervals: [start,color], [end,color] for each band
                        for idx, col in enumerate(colors_for_codes):
                            start = idx / n_codes
                            end = (idx + 1) / n_codes
                            colorscale.append([start, col])
                            colorscale.append([end, col])

                    heatmap = go.Figure(
                        data=go.Heatmap(
                            z=z,
                            x=x_vals,
                            y=y_vals,
                            colorscale=colorscale,
                            colorbar=dict(
                                title="Legend",
                                tickmode="array",
                                tickvals=list(party_codes.values()),
                                ticktext=[code_to_party[i] for i in sorted(code_to_party.keys())],
                            ),
                            hoverinfo="text",
                            text=hovertext,
                            zmin=0,
                            zmax=max(0, n_codes - 1)
                        )
                    )

                    heatmap.update_layout(
                        title="Winning party under different simulations",
                        xaxis_title="% non-voters convinced to vote for Your Party",
                        yaxis_title="% left voters converted to Your Party",
                        height=480,
                        margin=dict(l=60, r=40, t=60, b=60),
                    )

                    heatmap.update_xaxes(tickmode="linear", dtick=5)
                    heatmap.update_yaxes(tickmode="linear", dtick=5, autorange="reversed")

                    # Force rendering inside the expander (use_container_width True so it fills the expander)
                    st.plotly_chart(heatmap, width='content', config={"displayModeBar": False})

# If user selected a constituency from the sidebar, add an annotation on the map
if selected_constituency:
    sel_slug = slug_name(selected_constituency)
    centroid = find_centroid_by_slug(slug_to_centroid, sel_slug)
    if centroid is not None:
        cx, cy = centroid
        fig.update_layout(annotations=[dict(x=cx, y=cy, text=f"<b>{selected_constituency}</b>", showarrow=True, arrowhead=2, ax=20, ay=-20)])

st.markdown("To get in contact and request additional analysis, contact jessica.rapson@stats.oc.ac.uk", unsafe_allow_html=False, *, help=None, width="stretch")
