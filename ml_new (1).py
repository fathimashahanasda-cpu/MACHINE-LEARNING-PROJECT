import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained pipeline
pipeline = joblib.load("full_pipeline.pkl")

# Load sample data for dropdowns
df = pd.read_csv("transactions-2025-05-19.csv")

# Streamlit UI
st.title("🏠 Dubai Property Type Classifier")
st.markdown("""
Predict whether a real estate transaction is for **Land**, **Building**, or **Unit**
based on Dubai Land Department data.
""")

# Inputs
area = st.number_input("Property Area (square meter)", min_value=0.0)
value = st.number_input("Transaction Value (AED)", min_value=0.0)
parking = st.number_input("Parking Spaces", min_value=0)
rooms = st.selectbox("Number of Bedrooms", ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", 
                                            "5 B/R", "6 B/R", "7 B/R", "8 B/R", "9 B/R", "10 B/R"])
offplan = st.selectbox("Is Offplan?", ["Yes", "No"])
freehold = st.selectbox("Is Freehold?", ["Free Hold", "Non Free Hold"])
metro = st.checkbox("Near Metro")
mall = st.checkbox("Near Mall")
landmark = st.checkbox("Near Landmark")
project = st.checkbox("Has Project")

procedure_options = df['PROCEDURE_EN'].dropna().unique()
procedure_type = st.selectbox("Procedure Type", sorted(procedure_options))

area_options = df['AREA_EN'].dropna().unique()
area_name = st.selectbox("Area Name", sorted(area_options))

subtype_options = df['PROP_SB_TYPE_EN'].dropna().unique()
sub_type = st.selectbox("Sub Property Type", sorted(subtype_options))

room_mapping = {"Studio": 0, "1 B/R": 1, "2 B/R": 2, "3 B/R": 3, "4 B/R": 4,
                "5 B/R": 5, "6 B/R": 6, "7 B/R": 7, "8 B/R": 8, "9 B/R": 9, "10 B/R": 10}

if st.button("Predict Property Type"):
    input_data = {
        "ACTUAL_AREA": [area],
        "AREA_EN": [area_name],
        "TRANS_VALUE": [value],
        "PARKING": [parking],
        "ROOMS_EN": [room_mapping[rooms]],
        "IS_OFFPLAN_EN": [1 if offplan == "Yes" else 0],
        "IS_FREE_HOLD_EN": [1 if freehold == "Free Hold" else 0],
        "NEAREST_METRO_EN": [int(metro)],
        "NEAREST_MALL_EN": [int(mall)],
        "NEAREST_LANDMARK_EN": [int(landmark)],
        "PROJECT_EN": [int(project)],
        "PROCEDURE_EN": [procedure_type],
        "PROP_SB_TYPE_EN": [sub_type]
    }

    df_input = pd.DataFrame(input_data)

    # Predict using the full pipeline
    prediction = pipeline.predict(df_input)[0]
    class_map = {0: "Unit", 1: "Building", 2: "Land"}
    st.success(f"🏷 Predicted Property Type: *{class_map.get(prediction, 'Unknown')}*")
