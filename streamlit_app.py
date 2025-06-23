import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load model dan scaler
model = joblib.load('model_rf_best.joblib')
scaler = joblib.load('scaler.joblib')  # Pastikan kamu sudah menyimpan scaler juga

st.title("🎓 Dropout Prediction App")
st.markdown("Masukkan data siswa di bawah ini untuk memprediksi apakah siswa akan *dropout* atau *graduate*.")

# Inputan fitur
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Enrollment", min_value=16, max_value=70, value=18)
    application_mode = st.selectbox("Application Mode", options=[1, 17, 18, 39], format_func=lambda x: f"Mode {x}")
    course = st.selectbox("Course", options=[33, 171, 8014])
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    marital_status = st.selectbox("Marital Status", options=[1, 2, 3, 4, 5, 6])
    previous_qualification = st.number_input("Previous Qualification", min_value=1, max_value=20, value=1)
    scholarship = st.checkbox("Scholarship Holder", value=False)

with col2:
    curricular_enrolled_1st = st.number_input("Units Enrolled 1st Sem", min_value=0, max_value=30)
    curricular_approved_1st = st.number_input("Units Approved 1st Sem", min_value=0, max_value=30)
    curricular_grade_1st = st.number_input("Grade 1st Sem", min_value=0.0, max_value=20.0)
    curricular_enrolled_2nd = st.number_input("Units Enrolled 2nd Sem", min_value=0, max_value=30)
    curricular_approved_2nd = st.number_input("Units Approved 2nd Sem", min_value=0, max_value=30)
    curricular_grade_2nd = st.number_input("Grade 2nd Sem", min_value=0.0, max_value=20.0)
    tuition_paid = st.checkbox("Tuition Paid", value=True)
    debtor = st.checkbox("Debtor", value=False)

# Buat dataframe dari input
input_dict = {
    'Age_at_enrollment': age,
    'Application_mode': application_mode,
    'Course': course,
    'Gender': gender,
    'Marital_status': marital_status,
    'Previous_qualification': previous_qualification,
    'Scholarship_holder': int(scholarship),
    'Tuition_fees_up_to_date': int(tuition_paid),
    'Debtor': int(debtor),
    'Curricular_units_1st_sem_enrolled': curricular_enrolled_1st,
    'Curricular_units_1st_sem_approved': curricular_approved_1st,
    'Curricular_units_1st_sem_grade': curricular_grade_1st,
    'Curricular_units_2nd_sem_enrolled': curricular_enrolled_2nd,
    'Curricular_units_2nd_sem_approved': curricular_approved_2nd,
    'Curricular_units_2nd_sem_grade': curricular_grade_2nd
}

input_df = pd.DataFrame([input_dict])

# Scaling
scaled_input = scaler.transform(input_df)

# Prediksi
if st.button("Predict Dropout Status"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]  # Probabilitas dropout

    if pred == 1:
        st.error(f"❌ Siswa diprediksi akan Dropout (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Siswa diprediksi akan Graduate (Probabilitas Dropout: {prob:.2f})")
