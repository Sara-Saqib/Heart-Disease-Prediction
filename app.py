import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("❤️ Heart Disease Prediction System")
st.caption("AIC354 Machine Learning Fundamentals - educational demo, not a medical diagnosis.")

CP_LABELS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic",
}

age = st.number_input("Age (years)", min_value=29, max_value=77, value=55, step=1)
cp = st.selectbox("Chest pain type", options=list(CP_LABELS), format_func=CP_LABELS.get, index=3)
thalach = st.number_input("Maximum heart rate achieved (bpm)", min_value=71, max_value=202, value=150, step=1)
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)

if st.button("Predict"):
    # same column names/order as the training inputs; the scaler is inside the saved pipeline
    features = pd.DataFrame({"age": [age], "cp": [cp], "thalach": [thalach], "oldpeak": [oldpeak]})
    result = model.predict(features)[0]
    if result == 1:
        st.error("Prediction: HEART DISEASE PRESENT")
    else:
        st.success("Prediction: NO HEART DISEASE")
