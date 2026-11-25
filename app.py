"""Streamlit dashboard for the educational fall-detection prototype."""
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from train_model import make_demo_data, FEATURES
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Elderly Fall Detection", page_icon="🧓", layout="centered")
st.title("🧓 Elderly Fall Detection")
st.caption("Machine-learning demonstration using synthetic sensor-like readings.")

MODEL_PATH = Path(__file__).parent / "fall_model.joblib"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)["model"]
    X, y = make_demo_data()
    return RandomForestClassifier(n_estimators=160, max_depth=8, random_state=42,
                                  class_weight="balanced").fit(X, y)

model = load_model()
st.info("Demo only: no physical sensor is connected. Do not use this app for real-world safety monitoring.")

with st.sidebar:
    st.header("Sensor-like inputs")
    acceleration = st.slider("Acceleration magnitude (g)", 0.0, 5.0, 1.0, 0.05)
    gyroscope = st.slider("Gyroscope magnitude (deg/s)", 0.0, 500.0, 35.0, 1.0)
    posture = st.slider("Posture angle (degrees)", 0.0, 180.0, 18.0, 1.0)
    if st.button("Load sample fall reading"):
        st.session_state["sample_fall"] = True
    if st.session_state.get("sample_fall"):
        st.warning("Sample values: acceleration 2.8 g, gyroscope 260 deg/s, posture 82°")
        if st.button("Apply sample values"):
            st.session_state.update(accel=2.8, gyro=260.0, angle=82.0)

# Sliders remain the authoritative inputs; sample values are also available below.
st.subheader("Prediction")
manual = st.checkbox("Use sample fall values", value=False)
values = [2.8, 260.0, 82.0] if manual else [acceleration, gyroscope, posture]
prediction = int(model.predict([values])[0])
probability = model.predict_proba([values])[0]
fall_probability = float(probability[1])
if prediction:
    st.error(f"Model result: FALL-like pattern ({fall_probability:.1%} demo probability)")
else:
    st.success(f"Model result: NORMAL-like pattern ({fall_probability:.1%} demo probability)")
st.progress(fall_probability)
st.caption("The probability is a model score from synthetic training data, not a calibrated real-world risk estimate.")

with st.expander("About this prototype"):
    st.write("A Random Forest model learns patterns from generated example readings. Real deployment would require validated datasets, wearable-device integration, clinical/safety evaluation, and a reliable alert workflow.")
