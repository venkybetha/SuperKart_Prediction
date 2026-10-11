import os
import streamlit as st
import pandas as pd
import joblib

# Load the SuperKart regression model
model_path = os.path.join(os.path.dirname(__file__), "best_superkart_sales_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for SuperKart Sales Forecasting
st.title("SuperKart Store Sales Forecasting")
st.write("Enter product and store attributes to forecast total sales revenue.")

# Numerical Inputs
Product_Weight = st.number_input("Product Weight", min_value=0.1, max_value=100.0, value=12.0)
Product_Allocated_Area = st.slider("Product Allocated Display Area Ratio", 0.0, 1.0, 0.05, step=0.001)
Product_MRP = st.number_input("Product Maximum Retail Price (MRP)", min_value=1.0, max_value=1000.0, value=140.0)
Store_Establishment_Year = st.slider("Store Establishment Year", 1980, 2026, 1999)

# Categorical Inputs
Product_Id = st.text_input("Product ID", value="FD6114")
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Type = st.selectbox("Product Category", [
    "Snack Foods", "Fruits and Vegetables", "Household", "Frozen Foods", 
    "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Soft Drinks", 
    "Meat", "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"
])
Store_Size = st.selectbox("Store Size", ["Medium", "Small", "High"])
Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
])

# Prepare input dataframe matching training feature structure
input_data = pd.DataFrame([{
    'Product_Id': Product_Id,
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict button
if st.button("Forecast Sales"):
    forecast = model.predict(input_data)[0]
    st.success(f"Forecasted Total Sales Revenue: ${forecast:,.2f}")
