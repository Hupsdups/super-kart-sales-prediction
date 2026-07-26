
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Frontend Streamlit App
st.set_page_config(page_title="Super Kart Sales Predictor", layout="wide")

st.title('Super Kart Sales Prediction App')

# Backend API URL (this will be 'http://backend:7860' in a Docker network)
BACKEND_URL = 'http://backend:7860/predict'

st.write("Enter the details below to predict the product sales for a given store.")

# Input fields for product details
st.header('Product Details')
product_weight = st.number_input('Product Weight', min_value=4.0, max_value=22.0, value=12.0, step=0.1)
product_sugar_content = st.selectbox('Product Sugar Content', ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.number_input('Product Allocated Area', min_value=0.004, max_value=0.298, value=0.05, step=0.001, format="%.3f")
product_type = st.selectbox('Product Type', [
    'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Household',
    'Snack Foods', 'Meat', 'Hard Drinks', 'Fruits and Vegetables', 'Breads', 'Soft Drinks',
    'Breakfast', 'Starchy Foods', 'Seafood', 'Others'
])
product_mrp = st.number_input('Product MRP (Max Retail Price)', min_value=31.0, max_value=266.0, value=150.0, step=0.1)

# Input fields for store details
st.header('Store Details')
store_establishment_year = st.number_input('Store Establishment Year', min_value=1987, max_value=datetime.now().year, value=2000, step=1)
store_size = st.selectbox('Store Size', ['Small', 'Medium', 'High'])
store_location_city_type = st.selectbox('Store Location City Type', ['Tier 1', 'Tier 2', 'Tier 3'])
store_type = st.selectbox('Store Type', ['Supermarket Type1', 'Departmental Store', 'Supermarket Type2', 'Food Mart'])

# Predict button
if st.button('Predict Sales'):
    # Create a dictionary of input features
    input_data = {
        'Product_Weight': product_weight,
        'Product_Sugar_Content': product_sugar_content,
        'Product_Allocated_Area': product_allocated_area,
        'Product_Type': product_type,
        'Product_MRP': product_mrp,
        'Store_Establishment_Year': store_establishment_year,
        'Store_Size': store_size,
        'Store_Location_City_Type': store_location_city_type,
        'Store_Type': store_type
    }

    # Convert to DataFrame for consistent handling (even for single row)
    input_df = pd.DataFrame([input_data])

    try:
        # Send a POST request to the Flask API
        response = requests.post(BACKEND_URL, json=input_df.to_dict(orient='records'))

        if response.status_code == 200:
            prediction = response.json().get('prediction')
            if prediction:
                st.success(f"Predicted Product Store Sales Total: ${prediction[0]:,.2f}")
            else:
                st.error("Prediction result not found in response.")
        else:
            st.error(f"Error from backend: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the backend is running and accessible.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
