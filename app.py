# app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def generate_sample_csv(path: Path):
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2025-09-01", freq="MS")
    regions = ["North", "South", "East", "West"]
    products = ["Product A", "Product B", "Product C", "Product D"]
    rows = []
    for i, date in enumerate(dates):
        base = 10000 + i * 250 + 1800 * np.sin(2 * np.pi * (i % 12) / 12)
        for r_idx, region in enumerate(regions):
            region_factor = 0.85 + 0.1 * r_idx + np.random.uniform(-0.05, 0.05)
            for p_idx, product in enumerate(products):
                product_factor = 0.9 + 0.08 * p_idx + np.random.uniform(-0.03, 0.03)
                mean_sales = base * region_factor * product_factor
                sales = float(np.round(np.random.normal(loc=mean_sales, scale=mean_sales * 0.12), 2))
                sales = max(sales, 0.0)
                margin = 0.07 + 0.03 * p_idx + np.random.normal(0, 0.02)
                profit = float(np.round(max(sales * margin, 0.0), 2))
                rows.append({"Date": date, "Region": region, "Product": product, "Sales": sales, "Profit": profit})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

DATA_PATH = Path("retail_sales_sample.csv")

# Load data (or generate if missing)
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
else:
    st.info("No retail_sales_sample.csv found — generating a sample dataset for demo.")
    df = generate_sample_csv(DATA_PATH)

# Sidebar filters
st.sidebar.header("Filters")
min_date = df["Date"].min()
max_date = df["Date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

all_regions = ["All"] + sorted(df["Region"].unique().tolist())
region_selected = st.sidebar.multiselect("Region", options=all_regions, default=["All"])

all_products = ["All"] + sorted(df["Product"].unique().tolist())
product_selected = st.sidebar.multiselect("Product", options=all_products, default=["All"])

# Apply filters
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)

if "All" not in region_selected and len(region_selected) > 0:
    mask &= df["Region"].isin(region_selected)
if "All" not in product_selected and len(product_selected) > 0:
    mask &= df["Product"].isin(product_selected)

df_f = df.loc[mask].copy()
if df_f.empty:
    st.warning("No data for the selected filters — please expand your date range or select different filters.")
    st.stop()

# Header + key metrics
total_sales = df_f["Sales"].sum()
total_profit = df_f["Profit"].sum()
avg_margin = total_profit / total_sales if total_sales else 0

col_kpi1, col_kpi2, col_kpi3 = st.columns([1,1,1])
col_kpi1.metric("Total Sales", f"${total_sales:,.0f}")
col_kpi2.metric("Total Profit", f"${total_profit:,.0f}")
col_kpi3.metric("Avg Profit Margin", f"{avg_margin:.2%}")

st.markdown("---")

# Prepare chart data
monthly_sales = df_f.groupby("Date", as_index=False)["Sales"].sum()
region_sales = df_f.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
product_sales = df_f.groupby("Product", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
corr = df_f[["Sales", "Profit"]].corr()

# Layout: 2x2 grid
left_col, right_col = st.columns(2)

with left_col:
    # Top-left: Line chart (Sales over time)
    st.subheader("Sales Over Time")
    fig_line = px.line(monthly_sales, x="Date", y="Sales", markers=True, title="")
    fig_line.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=350)
    st.plotly_chart(fig_line, use_container_width=True)

    # Bottom-left: Heatmap (Sales vs Profit correlation)
    st.subheader("Sales ↔ Profit Correlation")
    heat_z = corr.values
    labels = corr.columns.tolist()
    fig
