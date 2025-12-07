import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Menentukan Happines Indeks",
    page_icon=":tangerine:"
)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'model_happines_indeks')

# Load model with error handling
model = None
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model from: {model_path}")
    st.exception(e)
    st.stop()

# Design Streamlit
st.title("Prediksi Tingkat Kebahagiaan Berdasarkan Gaya Hidup dan Penggunaan Media Sosial")
st.markdown(
    "Aplikasi ini memprediksi tingkat kebahagiaan seseorang berdasarkan faktor-faktor seperti usia, jenis kelamin, waktu layar harian, kualitas tidur, tingkat stres, hari tanpa media sosial, frekuensi olahraga, dan platform media sosial yang digunakan."
)

# Inputs (ensure these column names and order match the training script)
Age = st.slider("Umur", 10, 80, 25)
Gender = st.radio("Jenis Kelamin", ["Male", "Female"], index=0)
Daily_Screen_Time = st.slider("Waktu Layar Harian (jam)", 0.0, 24.0, 5.0)
Sleep_Quality = st.slider("Kualitas Tidur (1-10)", 1.0, 10.0, 7.0)
Stress_Level = st.slider("Tingkat Stres (1-10)", 1.0, 10.0, 5.0)
Days_Without_Social_Media = st.slider("Hari Tanpa Media Sosial", 0, 30, 3)
Exercise_Frequency = st.slider("Frekuensi Olahraga (minggu)", 0, 14, 4)
Social_Media_Platform = st.selectbox("Platform Media Sosial", ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter", "YouTube"], index=0)

if st.button("Prediksi Tingkat Kebahagiaan"):
    # Build DataFrame with explicit columns (must match training)
    cols = [
        'Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)',
        'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)',
        'Social_Media_Platform'
    ]
    data_baru = pd.DataFrame([[Age, Gender, Daily_Screen_Time, Sleep_Quality, Stress_Level, Days_Without_Social_Media, Exercise_Frequency, Social_Media_Platform]], columns=cols)

    try:
        # Predict (model is a Pipeline that includes preprocessing)
        prediksi = model.predict(data_baru)[0]

        # Try to obtain probability/confidence if available
        confidence = None
        try:
            proba = model.predict_proba(data_baru)
            if proba is not None:
                confidence = max(proba[0])
        except Exception:
            confidence = None

        if confidence is not None:
            st.success(f"Tingkat Kebahagiaan yang diprediksi adalah: **{prediksi}** dengan tingkat keyakinan **{confidence*100:.2f}%**")
        else:
            st.success(f"Tingkat Kebahagiaan yang diprediksi adalah: **{prediksi}**")

    except Exception as e:
        st.error("Prediction failed — check input features and model compatibility.")
        st.exception(e)

st.divider()
st.caption("@2025 Hana Rohadah")
st.balloons()
