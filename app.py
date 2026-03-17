import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px # Better for interactive charts

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# DATA & MODEL LOADING (With Error Handling)
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("housing_dataset_5000.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please ensure 'housing_dataset_5000.csv' is in the directory.")
        return None

@st.cache_resource
def load_model():
    try:
        return joblib.load("house_price_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'house_price_model.pkl' is uploaded.")
        return None

dataset = load_data()
model = load_model()

# -----------------------------
# SIDEBAR - INPUTS
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=100)
    st.header("🏡 Property Details")
    st.info("Adjust the sliders to see the estimated market value.")
    
    area = st.slider("Total Area (sq ft)", 500, 5000, 2000, step=50)
    bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4], index=1)
    stories = st.radio("Number of Stories", [1, 2, 3, 4], horizontal=True)
    parking = st.segmented_control("Parking Spaces", [0, 1, 2, 3], default=1)

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
if dataset is not None and model is not None:
    st.title("💰 House Price Prediction Dashboard")
    
    # Prediction Logic
    input_features = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_features)[0]

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Estimated Price", value=f"₹ {int(prediction):,}")
    with m2:
        st.metric(label="Price per Sq.Ft", value=f"₹ {int(prediction/area):,}")
    with m3:
        st.metric(label="Model Type", value="Decision Tree")

    st.divider()

    # Visualizations
    st.subheader("📊 Market Insights")
    tab1, tab2 = st.tabs(["Area vs Price", "Model Accuracy"])

    with tab1:
        # Interactive Plotly Chart
        fig_area = px.scatter(
            dataset.sample(1000), 
            x="area", 
            y="price", 
            color="price",
            title="Relationship: House Area vs. Price",
            labels={"area": "Area (sq ft)", "price": "Price (₹)"},
            template="plotly_white",
            color_continuous_scale="Viridis"
        )
        # Add a dot for the current prediction
        fig_area.add_scatter(x=[area], y=[prediction], mode='markers', 
                            marker=dict(size=15, color='red', symbol='star'),
                            name="Your Prediction")
        st.plotly_chart(fig_area, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns([1, 2])
        
        with col_a:
            importance = pd.DataFrame({
                'Feature': ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"],
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Feature Impact")
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col_b:
            sample = dataset.sample(min(300, len(dataset)))
            y_true = sample.iloc[:, -1]
            y_pred = model.predict(sample.iloc[:, :-1])
            
            fig_acc = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                               title="Actual vs. Predicted (Sample)", opacity=0.5)
            fig_acc.add_shape(type="line", x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max(),
                             line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig_acc, use_container_width=True)

else:
    st.warning("Please upload your dataset and model files to enable the dashboard.")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Developed by Chinmay • Built with Streamlit, Scikit-Learn, and Plotly")
