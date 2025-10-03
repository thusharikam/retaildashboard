# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Retail Sales Dashboard")
df = pd.read_csv("retail_sales_sample.csv", parse_dates=['Date'])
st.markdown("### Raw data")
st.dataframe(df.head(200))

# Sales trend
monthly = df.groupby('Date', as_index=False)['Sales'].sum()
fig1 = plt.figure(figsize=(10,4))
plt.plot(monthly['Date'], monthly['Sales'], marker='o')
plt.title('Total Sales Over Time'); plt.xlabel('Date'); plt.ylabel('Sales'); plt.xticks(rotation=45)
st.pyplot(fig1)

# Sales by region
region = df.groupby('Region', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False)
fig2 = plt.figure(figsize=(8,4))
plt.bar(region['Region'], region['Sales'])
plt.title('Total Sales by Region'); plt.xlabel('Region'); plt.ylabel('Sales')
st.pyplot(fig2)

# Pie: sales by product
product = df.groupby('Product', as_index=False)['Sales'].sum()
fig3 = plt.figure(figsize=(6,6))
plt.pie(product['Sales'], labels=product['Product'], autopct='%1.1f%%', startangle=90)
plt.title('Sales Contribution by Product')
st.pyplot(fig3)
