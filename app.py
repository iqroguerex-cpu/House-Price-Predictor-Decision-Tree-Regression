import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("housing_dataset_5000.csv")

dataset = load_data()

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Prediction Dashboard")
st.markdown("Predict house prices using Machine Learning")

st.write("---")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area (sq ft)", 500, 4000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 3, 2)
stories = st.sidebar.slider("Stories", 1, 2, 1)
parking = st.sidebar.slider("Parking", 0, 2, 1)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
prediction = model.predict(input_data)

st.subheader("💰 Predicted Price")
st.success(f"₹ {int(prediction[0]):,}")

st.write("---")

# -----------------------------
# CHARTS (OPTIMIZED)
# -----------------------------
col1, col2 = st.columns(2)

# 📊 Area vs Price
with col1:
    st.subheader("📊 Area vs Price")
    fig1, ax1 = plt.subplots()
    ax1.scatter(dataset["area"], dataset["price"])
    ax1.set_xlabel("Area")
    ax1.set_ylabel("Price")
    st.pyplot(fig1)

# 📈 Prediction vs Actual (SAMPLED for speed)
with col2:
    st.subheader("📈 Prediction vs Actual")

    sample = dataset.sample(500, random_state=1)  # 🔥 key fix
    X_sample = sample.iloc[:, :-1]
    y_sample = sample.iloc[:, -1]

    y_pred_sample = model.predict(X_sample)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_sample, y_pred_sample)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("🌳 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig3, ax3 = plt.subplots()
ax3.bar(features, importance)
plt.xticks(rotation=45)

st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
