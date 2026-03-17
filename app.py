import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Pro",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# ASSET LOADING
# -----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("housing_dataset_5000.csv")
    except:
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load("house_price_model.pkl")
    except:
        return None

dataset = load_data()
model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("🏠 Smart House Price Predictor")
st.markdown("Adjust the parameters in the sidebar and click **Predict House Price** to get started.")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📍 Property Features")
    
    area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=2000, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 2)
    stories = st.selectbox("Stories", [1, 2, 3, 4], index=0)
    parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)
    
    st.divider()
    # THE PREDICT BUTTON
    predict_btn = st.button("🚀 Predict House Price", use_container_width=True, type="primary")

# -----------------------------
# MAIN DASHBOARD LOGIC
# -----------------------------
if dataset is not None and model is not None:
    
    # Create a placeholder for the result
    result_container = st.container()

    if predict_btn:
        with st.spinner('Calculating market value...'):
            time.sleep(0.5) # Adding a tiny delay for "feel"
            
            input_features = np.array([[area, bedrooms, bathrooms, stories, parking]])
            prediction = model.predict(input_features)[0]
            
            with result_container:
                st.balloons()
                st.success("### Prediction Complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated Market Value", f"₹ {int(prediction):,}")
                c2.metric("Value per Sq.Ft", f"₹ {int(prediction/area):,}")
                c3.metric("Property Size", f"{area} sq.ft")
                st.divider()
    else:
        result_container.info("Enter details in the sidebar and click the button to see the predicted price.")

    # -----------------------------
    # INTERACTIVE VISUALIZATIONS
    # -----------------------------
    st.subheader("📊 Market Trends & Analysis")
    
    tab1, tab2 = st.tabs(["💰 Price Distribution", "📈 Model Accuracy"])

    with tab1:
        # Interactive Scatter Plot
        fig = px.scatter(
            dataset.sample(min(1500, len(dataset))), 
            x="area", 
            y="price", 
            color="price",
            size="area",
            hover_data=['bedrooms', 'bathrooms'],
            title="Market Overview: Area vs Price",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"],
            'Impact': model.feature_importances_
        }).sort_values(by='Impact', ascending=True)
        
        fig_imp = px.bar(importance_df, x='Impact', y='Feature', orientation='h', 
                         title="Which feature affects price the most?")
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.error("Missing critical files (CSV or PKL). Please check your repository.")

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Built for Decision Tree Regression Analysis • © 2026")
