import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("housing_dataset_5000.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Load model
model = joblib.load("house_price_model.pkl")

# Page config
st.set_page_config(page_title="House Price Dashboard", layout="wide")

# Title
st.title("🏠 House Price Prediction Dashboard")

st.write("---")

# Sidebar inputs
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area", 500, 4000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 3, 2)
stories = st.sidebar.slider("Stories", 1, 2, 1)
parking = st.sidebar.slider("Parking", 0, 2, 1)

# Prediction
input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
prediction = model.predict(input_data)

st.subheader("💰 Predicted Price")
st.success(f"₹ {int(prediction[0]):,}")

st.write("---")

# 📊 CHART 1: Area vs Price
st.subheader("📊 Area vs Price")

fig1 = plt.figure()
plt.scatter(dataset["area"], dataset["price"])
plt.xlabel("Area")
plt.ylabel("Price")
st.pyplot(fig1)

# 📊 CHART 2: Prediction vs Actual
st.subheader("📈 Prediction vs Actual")

y_pred = model.predict(X)

fig2 = plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
st.pyplot(fig2)

# 📊 CHART 3: Feature Importance
st.subheader("🌳 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig3 = plt.figure()
plt.bar(features, importance)
plt.xticks(rotation=45)
st.pyplot(fig3)
