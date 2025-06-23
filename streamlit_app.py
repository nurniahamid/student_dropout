import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_rf_best.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.title("🎓 Dropout Prediction App")
st.markdown("Masukkan data siswa di bawah ini untuk memprediksi apakah siswa akan *dropout* atau *graduate*.")

# Input dari pengguna
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Enrollment", min_value=16, max_value=70, value=18)
    application_mode = st.selectbox("Application Mode", options=[1, 17, 18, 39])
    application_order = st.number_input("Application Order", min_value=1, max_value=10, value=1)
    course = st.selectbox("Course", options=[33, 171, 8014])
    attendance = st.selectbox("Daytime or Evening", options=[1, 0], format_func=lambda x: "Daytime" if x == 1 else "Evening")
    previous_qualification = st.number_input("Previous Qualification", min_value=1, max_value=20, value=1)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    marital_status = st.selectbox("Marital Status", options=[1, 2, 3, 4, 5, 6])
    scholarship = st.checkbox("Scholarship Holder", value=False)
    debtor = st.checkbox("Debtor", value=False)
    tuition_paid = st.checkbox("Tuition Fees Up To Date", value=True)
    displaced = st.checkbox("Displaced", value=False)

with col2:
    mothers_qual = st.selectbox("Mother's Qualification", options=range(0, 20))
    fathers_qual = st.selectbox("Father's Qualification", options=range(0, 20))
    mothers_occ = st.selectbox("Mother's Occupation", options=range(0, 30))
    fathers_occ = st.selectbox("Father's Occupation", options=range(0, 30))
    cu_1_credited = st.number_input("1st Sem: Credited", min_value=0, max_value=30)
    cu_1_enrolled = st.number_input("1st Sem: Enrolled", min_value=0, max_value=30)
    cu_1_evals = st.number_input("1st Sem: Evaluations", min_value=0, max_value=30)
    cu_1_approved = st.number_input("1st Sem: Approved", min_value=0, max_value=30)
    cu_1_grade = st.number_input("1st Sem: Grade", min_value=0.0, max_value=20.0)
    cu_1_wo_eval = st.number_input("1st Sem: Without Evaluations", min_value=0, max_value=30)
    cu_2_enrolled = st.number_input("2nd Sem: Enrolled", min_value=0, max_value=30)
    cu_2_evals = st.number_input("2nd Sem: Evaluations", min_value=0, max_value=30)
    cu_2_approved = st.number_input("2nd Sem: Approved", min_value=0, max_value=30)
    cu_2_grade = st.number_input("2nd Sem: Grade", min_value=0.0, max_value=20.0)
    cu_2_wo_eval = st.number_input("2nd Sem: Without Evaluations", min_value=0, max_value=30)
    unemployment_rate = st.slider("Unemployment Rate (%)", min_value=0.0, max_value=100.0, value=5.0)

# Buat dict input
input_dict = {
    'Age_at_enrollment': age,
    'Application_mode': application_mode,
    'Application_order': application_order,
    'Course': course,
    'Daytime_evening_attendance': attendance,
    'Previous_qualification': previous_qualification,
    'Gender': gender,
    'Debtor': int(debtor),
    'Scholarship_holder': int(scholarship),
    'Marital_status': marital_status,
    'Mothers_qualification': mothers_qual,
    'Fathers_qualification': fathers_qual,
    'Mothers_occupation': mothers_occ,
    'Fathers_occupation': fathers_occ,
    'Displaced': int(displaced),
    'Tuition_fees_up_to_date': int(tuition_paid),
    'Curricular_units_1st_sem_credited': cu_1_credited,
    'Curricular_units_1st_sem_enrolled': cu_1_enrolled,
    'Curricular_units_1st_sem_evaluations': cu_1_evals,
    'Curricular_units_1st_sem_approved': cu_1_approved,
    'Curricular_units_1st_sem_grade': cu_1_grade,
    'Curricular_units_1st_sem_without_evaluations': cu_1_wo_eval,
    'Curricular_units_2nd_sem_enrolled': cu_2_enrolled,
    'Curricular_units_2nd_sem_evaluations': cu_2_evals,
    'Curricular_units_2nd_sem_approved': cu_2_approved,
    'Curricular_units_2nd_sem_grade': cu_2_grade,
    'Curricular_units_2nd_sem_without_evaluations': cu_2_wo_eval,
    'Unemployment_rate': unemployment_rate
}

# Tambahkan fitur duplikat yang digunakan saat modeling
input_dict['Application_mode'] = application_mode
input_dict['Application_order'] = application_order
input_dict['Course'] = course
input_dict['Daytime_evening_attendance'] = attendance
input_dict['Previous_qualification'] = previous_qualification

# DataFrame dan urutkan kolom
input_df = pd.DataFrame([input_dict], columns=feature_columns)
input_df = input_df[[ # urutan sesuai selected_features
    'Age_at_enrollment',
    'Application_mode',
    'Application_order',
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Gender',
    'Debtor',
    'Scholarship_holder',
    'Marital_status',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Displaced',
    'Tuition_fees_up_to_date',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate'
]]

# Scaling
scaled_input = scaler.transform(input_df)

# Prediksi
if st.button("Predict Dropout Status"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    if pred == 1:
        st.error(f"❌ Siswa diprediksi akan Dropout (Probabilitas: {prob:.2f})")
    else:
        st.success(f"✅ Siswa diprediksi akan Graduate (Probabilitas Dropout: {prob:.2f})")
