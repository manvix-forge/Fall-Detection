"""Train a demonstration fall classifier using synthetic sensor-like data."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "fall_model.joblib"
FEATURES = ["acceleration_g", "gyroscope_dps", "posture_angle_deg"]

def make_demo_data(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    rows = []
    for label in labels:
        if label == 1:
            rows.append([rng.normal(2.4, 0.55), rng.normal(230, 65), rng.normal(78, 18)])
        else:
            rows.append([rng.normal(1.0, 0.25), rng.normal(35, 22), rng.normal(18, 16)])
    frame = pd.DataFrame(rows, columns=FEATURES).clip(lower=0)
    return frame, labels

def main():
    X, y = make_demo_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=160, max_depth=8, random_state=42,
                                   class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Demo holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, target_names=["Normal", "Fall"]))
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
