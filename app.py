
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.datasets import fetch_openml

st.title("Araç Yakıt Tüketimi (MPG) Tahmini - 4 Model Karşılaştırmalı")

@st.cache_data
def load_data():
    data = fetch_openml("autoMpg", version=1, as_frame=True)
    df = data.frame
    df = df.replace("?", np.nan)
    df["horsepower"] = df["horsepower"].astype(float)
    df = df.dropna()
    df = df.drop(columns=["name"])
    features = ["model_year", "origin", "acceleration"]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    y = df["mpg"].astype(float)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

model_name = st.selectbox("Model Seçin", ["Linear Regression", "Random Forest", "KNN", "SVR"])

model_year = st.slider("Model Yılı (1970 - 1982)", 1970, 1982, 1976)
origin_label = st.selectbox("Üretim Bölgesi", ["ABD", "Avrupa", "Japonya"])
origin_mapping = {"ABD": 1, "Avrupa": 2, "Japonya": 3}
origin = origin_mapping[origin_label]
acceleration = st.slider("0-60 mil/saat Hızlanma Süresi", 8.0, 24.8, 15.0)

model_year_norm = (model_year - 1970) / (1982 - 1970)
origin_norm = (origin - 1) / 2
acceleration_norm = (acceleration - 8.0) / (24.8 - 8.0)

input_data = np.array([[model_year_norm, origin_norm, acceleration_norm]])

if st.button("Tahmin Et"):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsRegressor()
    elif model_name == "SVR":
        model = SVR()
    else:
        st.error("Model bulunamadı.")
        st.stop()

    model.fit(X_train, y_train)
    prediction = model.predict(input_data)
    st.success(f"{model_name} ile Tahmini MPG: {prediction[0]:.2f}")
