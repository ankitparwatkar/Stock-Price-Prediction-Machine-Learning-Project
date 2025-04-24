import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('stock.pkl', 'rb') as file:
    model = pickle.load(file)

# App title with styling
st.set_page_config(page_title="Stock Price Predictions", page_icon="📈", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: green;'>📈 Stock Price Predictions 📊</h1>
    <h3 style='text-align: center;'>Unlock future stock trends with AI-powered predictions! 🚀</h3>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.image("116316-abstract-cyan-defocused-background-design.jpg", use_column_width=True)
st.sidebar.header("Prediction Inputs")

# Input fields with styling
st.sidebar.subheader("Enter Stock Metrics:")
Prev_Close = st.sidebar.number_input("Previous Close Price", min_value=0.0, format="%.2f")
Open = st.sidebar.number_input("Open Price", min_value=0.0, format="%.2f")
High = st.sidebar.number_input("High Price", min_value=0.0, format="%.2f")
Low = st.sidebar.number_input("Low Price", min_value=0.0, format="%.2f")
Last = st.sidebar.number_input("Last Price", min_value=0.0, format="%.2f")

# Prediction button
if st.sidebar.button("🔮 Predict Future Price"):
    result = model.predict([[Prev_Close, Open, High, Low, Last]])[0]
    st.success(f"📌 **Predicted Price: ₹{result:.2f}**")
    
# Footer
st.markdown(
    """
    <hr>
    <h5 style='text-align: center; color: gray;'>Made with ❤️ using Streamlit & ML</h5>
    """,
    unsafe_allow_html=True
)