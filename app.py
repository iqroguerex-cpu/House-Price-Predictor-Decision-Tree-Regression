import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("housing_dataset_5000.csv")

dataset = load_data()

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Prediction Dashboard")
st.write("Simple ML app to estimate house prices")

st.write("---")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Input Features")

area = st.sidebar.number_input("Area (sq ft)", 500, 4000, 2000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 3, 2)
stories = st.sidebar.number_input("Stories", 1, 2, 1)
parking = st.sidebar.number_input("Parking", 0, 2, 1)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
prediction = model.predict(input_data)

st.subheader("💰 Predicted Price")
st.success(f"₹ {int(prediction[0]):,}")

st.write("---")

# -----------------------------
# LIGHTWEIGHT CHARTS
# -----------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Chart 1: Area vs Price
with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(dataset["area"], dataset["price"])
    ax1.set_xlabel("Area")
    ax1.set_ylabel("Price")
    ax1.set_title("Area vs Price")
    st.pyplot(fig1)

# Chart 2: Feature Importance
with col2:
    importance = model.feature_importances_
    features = dataset.columns[:-1]

    fig2, ax2 = plt.subplots()
    ax2.bar(features, importance)
    ax2.set_title("Feature Importance")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# -----------------------------
# OPTIONAL: SAMPLE PREDICTION GRAPH (SAFE)
# -----------------------------
st.subheader("📈 Model Behavior (Sampled)")

sample = dataset.sample(300, random_state=1)

X_sample = sample.iloc[:, :-1]
y_sample = sample.iloc[:, -1]
y_pred = model.predict(X_sample)

fig3, ax3 = plt.subplots()
ax3.scatter(y_sample, y_pred)
ax3.set_xlabel("Actual Price")
ax3.set_ylabel("Predicted Price")
ax3.set_title("Prediction vs Actual")

st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.caption("Built with Streamlit • Decision Tree Regression")
