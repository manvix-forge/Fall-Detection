# Elderly Fall Detection System

A beginner-friendly prototype that classifies sensor-like movement readings as **Fall** or **Normal** and displays the result in a Streamlit dashboard.

## Features
- Train a Random Forest classifier on generated demonstration data.
- Enter acceleration, gyroscope, and posture values manually.
- Try a generated sample reading.
- View the prediction and a simple confidence score.

## Tech stack
Python, Streamlit, pandas, NumPy, scikit-learn, joblib.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Input features
The demo uses acceleration magnitude (g), gyroscope magnitude (deg/s), and posture angle (degrees). The training data is synthetic and intended only to demonstrate the ML workflow.

## Important limitations
This is an educational prototype, not a medical or safety-certified device. It does not connect to a physical wearable sensor, contact caregivers, or provide emergency response. Do not use it to monitor a real person or make safety-critical decisions.

## Project structure
- `app.py` — Streamlit interface
- `train_model.py` — creates demo data and trains the classifier
- `requirements.txt` — dependencies
