"""A simple UI for the model from Students_Dropout_Simple.ipynb."""
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "torch"

import joblib
import keras
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="Student Dropout", page_icon="🎓")


@st.cache_resource
def load_model_files():
    # Load the trained model and the exact preprocessing used in the notebook.
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent / "models"
    model = keras.models.load_model(folder / "students_model.keras", compile=False)
    preprocessor = joblib.load(folder / "students_preprocessor.joblib")
    labels = joblib.load(folder / "students_labels.joblib")
    return model, preprocessor, labels


model, preprocessor, labels = load_model_files()

st.title("🎓 Student Dropout")
st.write("Введіть дані студента, щоб передбачити результат навчання.")
st.caption("Навчальна демонстрація. Модель використовує результати обох семестрів.")

# Use only course codes known to the fitted encoder.
courses = [int(code) for code in preprocessor.named_transformers_["cat"].categories_[0]]
with st.form("student"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Вік під час вступу", min_value=17, max_value=100, value=20)
        admission = st.number_input("Вступний бал (0–200)", 0.0, 200.0, 148.0, 0.1)
        course = st.selectbox("Освітня програма (код Course)", courses, index=courses.index(9119))
        gender = st.selectbox("Стать", [0, 1], format_func=lambda x: "Жіноча" if x == 0 else "Чоловіча")
        scholarship = st.selectbox("Отримує стипендію", [1, 0], format_func=lambda x: "Так" if x else "Ні")
    with right:
        approved1 = st.number_input("Складено дисциплін — 1-й семестр", min_value=0, value=5)
        grade1 = st.number_input("Середня оцінка — 1-й семестр (0–20)", 0.0, 20.0, 13.5, 0.1)
        approved2 = st.number_input("Складено дисциплін — 2-й семестр", min_value=0, value=6)
        grade2 = st.number_input("Середня оцінка — 2-й семестр (0–20)", 0.0, 20.0, 14.0, 0.1)
        fees = st.selectbox("Оплату навчання внесено вчасно", [1, 0], format_func=lambda x: "Так" if x else "Ні")
    submitted = st.form_submit_button("Передбачити", type="primary")

if submitted:
    # Keep original feature names; transform without fitting on the new record.
    record = pd.DataFrame([{
        "Age at enrollment": age,
        "Admission grade": admission,
        "Curricular units 1st sem (approved)": approved1,
        "Curricular units 1st sem (grade)": grade1,
        "Curricular units 2nd sem (approved)": approved2,
        "Curricular units 2nd sem (grade)": grade2,
        "Course": course,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Tuition fees up to date": fees,
    }])
    inputs = preprocessor.transform(record).astype("float32")
    probabilities = model.predict(inputs, verbose=0)[0]
    prediction = labels.inverse_transform([int(probabilities.argmax())])[0]
    meanings = {"Dropout": "Відрахований", "Enrolled": "Продовжує навчання", "Graduate": "Випускник"}
    st.success(f"Прогноз: {prediction} — {meanings[prediction]}")
    for column, label, probability in zip(st.columns(3), labels.classes_, probabilities):
        column.metric(label, f"{probability:.1%}")
    st.caption("Відсотки — оцінки моделі, а не гарантія результату конкретного студента.")
