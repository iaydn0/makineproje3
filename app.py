
import streamlit as st
import numpy as np
import joblib

st.title("Araç Yakıt Tüketimi (MPG) Tahmini - .PKL Modelli")

model_name = st.selectbox("Model Seçin", ["Linear Regression", "Random Forest", "KNN", "SVR"])

model_files = {
    "Linear Regression": "linear_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "KNN": "knn_model.pkl",
    "SVR": "svr_model.pkl"
}

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
    try:
        model_path = model_files[model_name]
        model = joblib.load(model_path)
        prediction = model.predict(input_data)
        st.success(f"{model_name} ile Tahmini MPG: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
