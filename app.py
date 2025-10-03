# app.py (fixed - no undefined 'fig' variable)
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Try importing plotly and give a clear error if missing
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    st.error("Required package 'plotly' is not installed. Run: pip install plotly")
    st.stop()

@st.cache_data
def generate_sample_df():
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
    return pd.DataFrame(rows)

DATA_PATH = Path("retail_sales_sample.csv")

# Load or generate dataset
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    except Exception as e:
        st.error("Failed to read CSV. Generating sample dataset instead.")
        st.exception(e)
        df = generate_sample_df()
else:
    st.info("No CSV found — generating a sample dataset.")
    df = generate_sample_df()

# Ensure Date column exists and is datetime
if "Date" not in df.columns:
    st.error("Data must contain a 'Date' column.")
    st.stop()
df["Date"] = pd.to_datetime(df["Date"])

# Sidebar filters (convert to python date objects)
st.sidebar.header("Filters")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_input = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_input, (tuple, list)):
    start_date = pd.to_datetime(date_input[0])
    end_date = pd.to_datetime(date_input[1])
else:
    start_date = pd.to_datetime(date_input)
    end_date = start_date

regions = sorted(df["Region"].unique().tolist())
products = sorted(df["Product"].unique().tolist())

region_selected = st.sidebar.multiselect("Region (choose All or specific)", options=["All"] + regions, default=["All"])
product_selected = st.sidebar.multiselect("Product (choose All or specific)", options=["All"] + products, default=["All"])

# Apply filters
mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
if "All" not in region_selected and len(region_selected) > 0:
    mask &= df["Region"].isin(region_selected)
if "All" not in product_selected and len(product_selected) > 0:
    mask &= df["Product"].isin(product_selected)

df_f = df.loc[mask].copy()
if df_f.empty:
    st.warning("No data for the selected filters. Try expanding the date range or removing some filters.")
    st.stop()

# KPIs
total_sales = df_f["Sales"].sum()
total_profit = df_f["Profit"].sum()
avg_margin = (total_profit / total_sales) if total_sales else 0

k1, k2, k3 = st.columns(3)
k1.metric("Total Sales", f"${total_sales:,.0f}")
k2.metric("Total Profit", f"${total_profit:,.0f}")
k3.metric("Avg Profit Margin", f"{avg_margin:.2%}")

st.markdown("---")

# Chart data
monthly_sales = df_f.groupby("Date", as_index=False)["Sales"].sum()
region_sales = df_f.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
product_sales = df_f.groupby("Product", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
corr = df_f[["Sales", "Profit"]].corr()

# 2x2 layout - ensure each figure var is unique and defined
left, right = st.columns(2)

with left:
    st.subheader("Sales Over Time")
    fig_line = px.line(monthly_sales, x="Date", y="Sales", markers=True)
    fig_line.update_layout(height=360, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Sales ↔ Profit Correlation")
    labels = corr.columns.tolist()
    heat_z = corr.values
    fig_heat = go.Figure(data=go.Heatmap(z=heat_z, x=labels, y=labels, colorscale="Viridis", zmin=-1, zmax=1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            fig_heat.add_annotation(x=labels[j], y=labels[i], text=f"{heat_z[i,j]:.2f}", showarrow=False,
                                    font=dict(color="white" if abs(heat_z[i,j])>0.5 else "black"))
    fig_heat.update_layout(height=360, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

with right:
    st.subheader("Sales by Region")
    fig_bar = px.bar(region_sales, x="Region", y="Sales", text_auto=True)
    fig_bar.update_layout(height=360, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Sales by Product (Contribution)")
    fig_pie = px.pie(product_sales, names="Product", values="Sales", hole=0.35)
    fig_pie.update_layout(height=360, margin=dict(t=30,l=10,r=10,b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
with st.expander("Show raw filtered data"):
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True)
