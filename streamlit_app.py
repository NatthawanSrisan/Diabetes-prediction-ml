import streamlit as st
import pickle
import numpy as np

# dowload
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# titlie
st.title("🩺 Diabetes Prediction App")

# input fields 10 features 
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 22.5)
highbp = st.checkbox("Have high blood pressure?")
highchol = st.checkbox("Have high cholesterol?")
cholcheck = st.checkbox("Had cholesterol check?")
smoker = st.radio("Do you smoke?", ["Yes", "No"])
physactivity = st.radio("Do you exercise?", ["Yes", "No"])
sex = st.selectbox("Gender", ["Male", "Female"])
education = st.slider("Education level (1–6)", 1, 6, 3)
income = st.slider("Income level (1–8)", 1, 8, 4)

# input data to array 
input_data = np.array([[age, bmi,
                         1 if highbp else 0,
                         1 if highchol else 0,
                         1 if cholcheck else 0,
                         1 if smoker == "Yes" else 0,
                         1 if physactivity == "Yes" else 0,
                         1 if sex == "Male" else 0,
                         education,
                         income]])

# predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction result: {prediction[0]}")