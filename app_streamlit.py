import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title = "Menentukan Happines Indeks",
    page_icon = ":tangerine:"
)

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, 'model_happines_indeks')
model = joblib.load(model_path)

# Design Streamlit
st.title("Prediksi Tingkat Kebahagiaan Berdasarkan Gaya Hidup dan Penggunaan Media Sosial")
st.markdown("Aplikasi ini memprediksi tingkat kebahagiaan seseorang berdasarkan faktor-faktor seperti usia, jenis kelamin, waktu layar harian, kualitas tidur, tingkat stres, hari tanpa media sosial, frekuensi olahraga, dan platform media sosial yang digunakan.")

Age = st.slider("Umur", 10, 80, 25)
Gender = st.pills("Jenis Kelamin", ["Male", "Female", "Other"], default="Male")
Daily_Screen_Time = st.slider("Waktu Layar Harian (jam)", 0, 24, 5)
Sleep_Quality = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
Stress_Level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
Days_Without_Social_Media = st.slider("Hari Tanpa Media Sosial", 0, 30, 3)
Exercise_Frequency = st.slider("Frekuensi Olahraga (minggu)", 0, 14, 4)
Social_Media_Platform = st.pills("Platform Media Sosial",["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter", "YouTube"], default="Facebook")

if st.button("Prediksi Tingkat Kebahagiaan", type="primary"):
    data_baru = pd.DataFrame([[Age, Gender, Daily_Screen_Time, Sleep_Quality, Stress_Level, Days_Without_Social_Media, Exercise_Frequency, Social_Media_Platform]], ['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform'])
    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Tingkat Kebahagiaan yang diprediksi adalah: **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")

st.divider()
st.caption("@2025 Hana Rohadah")
st.balloons()