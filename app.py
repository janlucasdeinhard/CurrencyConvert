# app.py
# ---------------------------------------------------------
# Historical FX Value Explorer (pair columns in the CSV)
# ---------------------------------------------------------
# Install:
#   pip install streamlit pandas plotly
#
# Run:
#   streamlit run app.py
#
# Expected CSV (your format):
#   "DATE","TIME PERIOD","rate","From Currency","To Currency"
#   "1999-01-04","04 Jan 1999","1.1789","USD","EUR"
#   ...
#
# Assumption:
#   rate = (From Currency) per 1 (To Currency)
#   Example: rate=1.1789, From=USD, To=EUR => 1 EUR = 1.1789 USD
#
# Conversions:
#   To -> From  : multiply by rate
#   From -> To  : divide by rate

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Config
# ----------------------------
DEFAULT_CSV_PATH = "fx_rates.csv"
DEFAULT_START_DATE = dt.date(2022, 1, 1)  # <-- requested default start


# ----------------------------
# Helpers
# ----------------------------
def clean_col(col: str) -> str:
    return re.sub(r"\s+", " ", str(col)).strip().strip('"').strip("'")


def as_date(x: Any) -> dt.date:
    """Convert Streamlit/pandas/numpy date-ish values to a plain datetime.date."""
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    try:
        return pd.to_datetime(x).date()
    except Exception as e:
        raise ValueError(f"Could not convert to date: {x!r}") from e


def normalize_ccy(x: Any) -> str:
    return str(x).strip().strip('"').strip("'").upper()


@st.cache_data(show_spinner=False)
def load_fx_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads CSV with columns:
      DATE, TIME PERIOD, rate, From Currency, To Currency

    Returns a tidy df with columns:
      date (datetime64), rate (float), from_ccy (str), to_ccy (str)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at '{csv_path}'. "
            f"Put your file there or change the path in the sidebar."
        )

    df = pd.read_csv(path)

    # Clean headers
    df.columns = [clean_col(c) for c in df.columns]

    required = {"DATE", "From Currency", "To Currency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Identify rate column: any column not among the known metadata columns
    non_rate = {"DATE", "TIME PERIOD", "From Currency", "To Currency"}
    candidate_cols = [c for c in df.columns if c not in non_rate]
    if len(candidate_cols) != 1:
        raise ValueError(
            "Expected exactly ONE rate column besides DATE/TIME PERIOD/From Currency/To Currency.\n"
            f"Found candidates: {candidate_cols}\n"
            "If your file has different columns, paste the header row and I’ll adapt."
        )
    rate_col = candidate_cols[0]

    out = df[["DATE", rate_col, "From Currency", "To Currency"]].copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out[rate_col] = pd.to_numeric(out[rate_col], errors="coerce")
    out["From Currency"] = out["From Currency"].map(normalize_ccy)
    out["To Currency"] = out["To Currency"].map(normalize_ccy)

    out = out.dropna(subset=["DATE", rate_col, "From Currency", "To Currency"])
    out = out[out[rate_col] > 0].sort_values("DATE")

    out = out.rename(
        columns={
            "DATE": "date",
            rate_col: "rate",
            "From Currency": "from_ccy",
            "To Currency": "to_ccy",
        }
    )

    # Ensure stable dtype for comparisons later
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    return out


def compute_converted_series(df: pd.DataFrame, amount: float, direction: str) -> pd.DataFrame:
    """
    direction:
      - 'TO_TO_FROM': convert To Currency -> From Currency  (multiply)
      - 'FROM_TO_TO': convert From Currency -> To Currency  (divide)

    Assumption: rate = from_ccy per 1 to_ccy
    """
    out = df.copy()
    if direction == "TO_TO_FROM":
        out["converted_value"] = amount * out["rate"]
    elif direction == "FROM_TO_TO":
        out["converted_value"] = amount / out["rate"]
    else:
        raise ValueError("Invalid direction.")
    return out


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="FX History Converter", layout="wide")
st.title("💱 Historical FX Value Explorer")
st.caption(
    "Enter an amount, choose currencies, and see what it would have been worth over time "
    "based on your historical exchange-rate CSV."
)

with st.sidebar:
    st.header("Data source")
    csv_path = st.text_input("CSV path", value=DEFAULT_CSV_PATH)
    st.caption('Expected: DATE, TIME PERIOD (optional), rate, From Currency, To Currency')

try:
    fx = load_fx_csv(csv_path)
except Exception as e:
    st.error(str(e))
    st.stop()

pairs = fx[["from_ccy", "to_ccy"]].drop_duplicates().sort_values(["from_ccy", "to_ccy"])
from_list = sorted(pairs["from_ccy"].unique().tolist())

c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.6])

with c1:
    amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)

with c2:
    from_ccy = st.selectbox("From currency", from_list, index=0 if from_list else None)

valid_to = sorted(pairs[pairs["from_ccy"] == from_ccy]["to_ccy"].unique().tolist())
if not valid_to:
    valid_to = sorted(pairs["to_ccy"].unique().tolist())

with c3:
    to_ccy = st.selectbox("To currency", valid_to, index=0 if valid_to else None)

with c4:
    data_min_d = as_date(fx["date"].min())
    data_max_d = as_date(fx["date"].max())

    # Requested default: Jan 1, 2022 to present (bounded by available data)
    default_start = max(DEFAULT_START_DATE, data_min_d)
    default_end = data_max_d  # "present" in the dataset

    date_range = st.date_input(
        "Date range",
        value=(default_start, default_end),
        min_value=data_min_d,
        max_value=data_max_d,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range

    start_d = as_date(start_d)
    end_d = as_date(end_d)

start = pd.Timestamp(start_d)
end = pd.Timestamp(end_d)

# Filter pair + dates
df_pair = fx[(fx["from_ccy"] == from_ccy) & (fx["to_ccy"] == to_ccy)].copy()
df_pair = df_pair[(df_pair["date"] >= start) & (df_pair["date"] <= end)].copy()

if df_pair.empty:
    st.warning(f"No data for {from_ccy}/{to_ccy} in the selected date range.")
    st.stop()

# Direction selection
direction_label = st.radio(
    "Conversion direction",
    options=[f"{to_ccy} → {from_ccy} (multiply by rate)", f"{from_ccy} → {to_ccy} (divide by rate)"],
    horizontal=True,
    index=1,  # default: From -> To
)

direction = "TO_TO_FROM" if direction_label.startswith(f"{to_ccy} →") else "FROM_TO_TO"
df_out = compute_converted_series(df_pair, float(amount), direction)

out_ccy = from_ccy if direction == "TO_TO_FROM" else to_ccy
in_ccy = to_ccy if direction == "TO_TO_FROM" else from_ccy

# KPIs
k1, k2, k3, k4 = st.columns(4)
latest = df_out.iloc[-1]
first = df_out.iloc[0]
min_row = df_out.loc[df_out["converted_value"].idxmin()]
max_row = df_out.loc[df_out["converted_value"].idxmax()]

with k1:
    st.metric(f"Latest ({latest['date'].date()})", f"{latest['converted_value']:.4f} {out_ccy}")
with k2:
    st.metric(f"First ({first['date'].date()})", f"{first['converted_value']:.4f} {out_ccy}")
with k3:
    st.metric(f"Min ({min_row['date'].date()})", f"{min_row['converted_value']:.4f} {out_ccy}")
with k4:
    st.metric(f"Max ({max_row['date'].date()})", f"{max_row['converted_value']:.4f} {out_ccy}")

st.divider()

left, right = st.columns([1.35, 1.0])

with left:
    st.subheader("Converted value over time")

    current_value = float(latest["converted_value"])

    fig = px.line(
        df_out,
        x="date",
        y="converted_value",
        title=f"What {amount:.2f} {in_ccy} would have been worth in {out_ccy}",
        labels={"date": "Date", "converted_value": f"Value in {out_ccy}"},
    )

    # Bright yellow horizontal line highlighting the current value
    fig.add_hline(
        y=current_value,
        line_color="yellow",
        line_width=3,
        line_dash="solid",
        annotation_text=f"Current: {current_value:.2f} {out_ccy}",
        annotation_position="top left",
        annotation_font_color="black",
        annotation_bgcolor="yellow",
        annotation_bordercolor="black",
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Data preview")
    st.dataframe(df_out, use_container_width=True, height=420)

with st.expander("Download filtered results"):
    export = df_out.copy()
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.download_button(
        "Download CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"converted_{in_ccy}_to_{out_ccy}.csv",
        mime="text/csv",
    )
