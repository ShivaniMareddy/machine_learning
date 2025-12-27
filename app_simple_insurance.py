import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Simple Linear Regression - Insurance", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class="card">
    <h1>Simple Linear Regression</h1>
    <p>Predict <b>Insurance Charges</b> from <b>Age</b></p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["age", "charges"]].head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data
x = df[["age"]]
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Age vs Insurance Charges")

fig, ax = plt.subplots()
ax.scatter(df["age"], df["charges"], alpha=0.6)
ax.plot(df["age"], model.predict(scaler.transform(x)), color="red")
ax.set_xlabel("Age")
ax.set_ylabel("Charges")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R²", f"{r2:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Insurance Charges")

age = st.slider("Age", int(df.age.min()), int(df.age.max()), 30)
prediction = model.predict(scaler.transform([[age]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Charges: ₹ {prediction:,.2f}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
