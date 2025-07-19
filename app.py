import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)
features_path = os.path.join(os.path.dirname(__file__), "features.pkl")
feature_columns = joblib.load(features_path)





property_types = sorted([col.replace("type_", "") for col in feature_columns if col.startswith("type_")])
locations = sorted([col.replace("location_", "") for col in feature_columns if col.startswith("location_")])
amenity_columns = [col for col in feature_columns if col.startswith("has_") or "amenity" in col]

base_path = os.path.dirname(__file__)

ver01_path = os.path.join(base_path, "properties_ver01.csv")
final_path = os.path.join(base_path, "properties_final.csv")

df = pd.read_csv(ver01_path)
df_final = pd.read_csv(final_path)



def predict_price(model, feature_columns, property_type, location, area, bedrooms, bathrooms, selected_amenities):
    input_df = pd.DataFrame([0] * len(feature_columns), index=feature_columns).T

    input_df['area_of_apt'] = area
    input_df['no_of_bedrooms'] = bedrooms
    input_df['no_of_bathrooms'] = bathrooms

    # Add average meter price feature
    avg_meter_price = 57117.58
    input_df['avg_meter_price'] = area * avg_meter_price

    # One-hot encode type and location
    type_col = f"type_{property_type}"
    loc_col = f"location_{location}"

    if type_col in input_df.columns:
        input_df[type_col] = 1
    else:
        st.warning(f"⚠️ Property type '{property_type}' not seen in training data.")

    if loc_col in input_df.columns:
        input_df[loc_col] = 1
    else:
        st.warning(f"⚠️ Location '{location}' not seen in training data.")

    for amenity in selected_amenities:
        if amenity in input_df.columns:
            input_df[amenity] = 1

    prediction = model.predict(input_df)[0]
    return round(prediction, 2)


st.set_page_config(page_title="Property Price Predictor", layout="wide")
st.markdown("""
    <style>
        .banner-img {
            width: 100%;
            border-radius: 50px;
            margin-bottom: 10px;
        }
    </style>
    <img class="banner-img" src="https://media.cnn.com/api/v1/images/stellar/prod/240315111335-01-egypt-new-administrative-capital.jpg?c=original" alt="Banner">
""", unsafe_allow_html=True)

st.title("New Adminstrative Capital Properties Price Prediction App", anchor="top")
st.markdown("Predict property prices based on area, location, and features using a trained machine learning model.")

tab1, tab2 = st.tabs(["Price Prediction", "Analytics Dashboard"])

with tab1:
    st.header("Enter Property Details:")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sqm)", min_value=30, max_value=1500, step=10)
        if area < 30 or area > 1500:
            st.warning("Area should be between 30 and 1500 sqm.")
        if area < 50:
            st.warning("Area less than 50 sqm may not be suitable for residential properties.")  

        if area < 100:
            bedrooms = st.selectbox("Number of Bedrooms", [1, 2])
            bathrooms = st.selectbox("Number of Bathrooms", [1])
        elif area < 200:
            bedrooms = st.selectbox("Number of Bedrooms", [2, 3, 4])   
            bathrooms = st.selectbox("Number of Bathrooms", [2, 3])   
        elif area < 300:
            bedrooms = st.selectbox("Number of Bedrooms", [3, 4, 5])
            bathrooms = st.selectbox("Number of Bathrooms", [2, 3, 4])
        else:
            bedrooms = st.selectbox("Number of Bedrooms", [3, 4, 5, 6, 7])
            bathrooms = st.selectbox("Number of Bathrooms", [2, 3, 4, 5])

    with col2:
        property_type = st.selectbox("Property Type", property_types)
        location = st.selectbox("Location", locations)

        st.subheader("Amenities")
        selected_amenities = []
        with st.expander("Choose Amenities"):
            amenity_cols = st.columns(3)  
            for idx, amenity in enumerate(amenity_columns):
                with amenity_cols[idx % 3]:
                    if st.checkbox(amenity.replace("_", " ").capitalize()):
                        selected_amenities.append(amenity)

    if st.button("Predict Price"):
        if not property_type or not location:
            st.error("Please enter both property type and location.")
        else:
            price = predict_price(model, feature_columns, property_type, location, area, bedrooms, bathrooms, selected_amenities)
            st.success(f"Predicted Price: {price:,.0f} EGP")


with tab2:
    st.subheader("Dataset Overview")
    st.write("This dataset contains property listings in the New Administrative Capital, Egypt. It includes various features such as area, number of bedrooms and bathrooms, property type, location, and amenities.")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)

    
    with col1:
        
        st.subheader("Average Price per Square Meter")

        df_final['price'] = pd.to_numeric(df_final['price'], errors='coerce')
        df_final['area_of_apt'] = pd.to_numeric(df_final['area_of_apt'], errors='coerce')

        df_final = df_final[(df_final['area_of_apt'] > 0) & (df_final['price'].notna())]


        df_final['price_per_sqm'] = df_final['price'] / df_final['area_of_apt']
        avg_price_per_sqm = df_final['price_per_sqm'].mean()


        st.metric("Average Price per sqm", f"{avg_price_per_sqm:,.2f} EGP")

        df_final['price'] = pd.to_numeric(df_final['price'], errors='coerce')
        df_final = df_final.dropna(subset=['price'])
        min_price = df_final['price'].min()
        
        st.metric("Minimum Price", f"{min_price:,.0f} EGP")


    with col2:

        df_final['price'] = pd.to_numeric(df_final['price'], errors='coerce')

        df_final = df_final.dropna(subset=['price'])


        avg_price = df_final['price'].mean()

        st.subheader("Average Apartment Price")
        st.metric("Average Price", f"{avg_price:,.0f} EGP")
        max_price = df_final['price'].max()
        st.metric("Maximum Price", f"{max_price:,.0f} EGP")

    

    
    
    st.subheader("Average Price by Location")
    avg_price_location = df.groupby("location")["price"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(avg_price_location)

    st.subheader("Average Price by Property Type")
    avg_price_type = df.groupby("type")["price"].mean().sort_values(ascending=False)
    st.bar_chart(avg_price_type)


    
