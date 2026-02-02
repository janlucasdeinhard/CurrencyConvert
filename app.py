# app.py
# ---------------------------------------------
# Historical FX Value Explorer (ECB-style CSV)
# ---------------------------------------------
# Install:
#   pip install streamlit pandas plotly
#
# Run (IMPORTANT):
#   streamlit run app.py
#
# CSV example (like yours):
#   DATE,"TIME PERIOD","US dollar/Euro "
#   1999-01-04,"04 Jan 1999","1.1789"
#
# Meaning (typically): "US dollar/Euro" = USD per 1 EUR
#   1 EUR = 1.1789 USD
# So:
#   EUR -> USD : multiply by rate
#   USD -> EUR : divide by rate

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

# Map human-readable names in header to ISO codes.
# Extend this as you encounter more files/currencies.
NAME_TO_ISO = {
    "US dollar": "USD",
    "Euro": "EUR",
    "Japanese yen": "JPY",
    "Pound sterling": "GBP",
    "Swiss franc": "CHF",
    "Canadian dollar": "CAD",
    "Australian dollar": "AUD",
    "Chinese yuan renminbi": "CNY",
}


# ----------------------------
# Helpers
# ----------------------------
def clean_col(col: str) -> str:
    # Normalize whitespace (handles embedded newlines) + strip quotes/spaces
    return re.sub(r"\s+", " ", str(col)).strip().strip('"').strip("'")


def parse_pair_from_header(header: str) -> Tuple[str, str]:
    """
    Header example: 'US dollar/Euro'
    Returns (base, quote) assuming header is 'BASE/QUOTE'
    and the rate is: BASE per 1 QUOTE (e.g., USD per EUR).
    """
    header = clean_col(header)
    if "/" not in header:
        raise ValueError(f"Could not parse currency pair from header: {header}")

    left, right = [p.strip() for p in header.split("/", 1)]
    base = NAME_TO_ISO.get(left, left.upper())
    quote = NAME_TO_ISO.get(right, right.upper())
    return base, quote


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


@st.cache_data(show_spinner=False)
def load_ecb_style_csv(csv_path: str) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Loads ECB-style CSV with columns:
      DATE, TIME PERIOD, <PAIR COLUMN>
    Returns:
      df with columns: date (datetime64), rate (float)
      base_ccy, quote_ccy, raw_rate_header
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found at '{csv_path}'. "
            f"Put your file there or change the path in the sidebar."
        )

    df = pd.read_csv(path)
    df.columns = [clean_col(c) for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError("Expected a 'DATE' column.")

    non_rate_cols = {"DATE", "TIME PERIOD"}
    rate_cols = [c for c in df.columns if c not in non_rate_cols]

    if len(rate_cols) != 1:
        raise ValueError(
            "Expected exactly ONE FX rate column besides DATE/TIME PERIOD.\n"
            f"Found: {rate_cols}\n"
            "If your file has multiple FX columns, tell me and I’ll adapt the app."
        )

    raw_rate_header = rate_cols[0]
    base, quote = parse_pair_from_header(raw_rate_header)

    out = df[["DATE", raw_rate_header]].copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out[raw_rate_header] = pd.to_numeric(out[raw_rate_header], errors="coerce")

    out = out.dropna(subset=["DATE", raw_rate_header])
    out = out[out[raw_rate_header] > 0].sort_values("DATE")

    out = out.rename(columns={"DATE": "date", raw_rate_header: "rate"})

    # Ensure dtype is truly datetime64 for clean comparisons later
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    return out, base, quote, raw_rate_header


def compute_converted_series(df: pd.DataFrame, amount: float, direction: str) -> pd.DataFrame:
    """
    direction:
      - 'QUOTE_TO_BASE' => QUOTE -> BASE using BASE/QUOTE rate: amount * rate
      - 'BASE_TO_QUOTE' => BASE -> QUOTE: amount / rate
    """
    out = df.copy()
    if direction == "QUOTE_TO_BASE":
        out["converted_value"] = amount * out["rate"]
    elif direction == "BASE_TO_QUOTE":
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
    "Enter an amount, pick direction, and see what it would have been worth over time "
    "based on the historical exchange rates in your ECB-style CSV."
)

with st.sidebar:
    st.header("Data source")
    csv_path = st.text_input("CSV path", value=DEFAULT_CSV_PATH)
    st.caption('Expected columns: DATE, "TIME PERIOD", "<pair like US dollar/Euro>"')

df, base, quote, raw_header = load_ecb_style_csv(csv_path)

st.caption(f"Detected pair column **{raw_header}** ⇒ rate = **{base} per 1 {quote}**")

c1, c2, c3 = st.columns([1.2, 1.2, 1.8])

with c1:
    amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)

with c2:
    direction_label = st.selectbox(
        "Convert",
        options=[f"{quote} → {base}", f"{base} → {quote}"],
        index=0,
    )
    direction = "QUOTE_TO_BASE" if direction_label.startswith(f"{quote} →") else "BASE_TO_QUOTE"

with c3:
    # Make sure min/max are plain python dates
    min_d = as_date(df["date"].min())
    max_d = as_date(df["date"].max())

    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )

    # Normalize to (start_d, end_d) as plain python dates
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = end_d = date_range

    start_d = as_date(start_d)
    end_d = as_date(end_d)

# Crucial: force SCALAR timestamps (not array/Series of len 1)
start = pd.Timestamp(start_d)
end = pd.Timestamp(end_d)

# Filter + compute
df_f = df[(df["date"] >= start) & (df["date"] <= end)].copy()
if df_f.empty:
    st.warning("No rows in this date range.")
    st.stop()

df_out = compute_converted_series(df_f, float(amount), direction)

to_ccy = base if direction == "QUOTE_TO_BASE" else quote
from_ccy = quote if direction == "QUOTE_TO_BASE" else base

# KPIs
k1, k2, k3, k4 = st.columns(4)

latest = df_out.iloc[-1]
first = df_out.iloc[0]
min_row = df_out.loc[df_out["converted_value"].idxmin()]
max_row = df_out.loc[df_out["converted_value"].idxmax()]

with k1:
    st.metric(f"Latest ({latest['date'].date()})", f"{latest['converted_value']:.4f} {to_ccy}")
with k2:
    st.metric(f"First ({first['date'].date()})", f"{first['converted_value']:.4f} {to_ccy}")
with k3:
    st.metric(f"Min ({min_row['date'].date()})", f"{min_row['converted_value']:.4f} {to_ccy}")
with k4:
    st.metric(f"Max ({max_row['date'].date()})", f"{max_row['converted_value']:.4f} {to_ccy}")

st.divider()

left, right = st.columns([1.35, 1.0])

with left:
    st.subheader("Converted value over time")
    current_value = latest["converted_value"]
    fig = px.line(
        df_out,
        x="date",
        y="converted_value",
        title=f"What {amount:.2f} {from_ccy} would have been worth in {to_ccy}",
        labels={"date": "Date", "converted_value": f"Value in {to_ccy}"},
    )
    fig.add_hline(
        y=current_value,
        line_color="yellow",
        line_width=3,
        line_dash="solid",
        annotation_text=f"Current value: {current_value:.2f} {to_ccy}",
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
    # Avoid .dt.date typing issues; produce clean ISO strings for CSV
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.download_button(
        "Download CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"converted_{from_ccy}_{to_ccy}.csv",
        mime="text/csv",
    )
