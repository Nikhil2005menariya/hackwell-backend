import pandas as pd
from models.obesity.obesity_model import load_models, preprocess_new_data, predict_patient

def predict_obesity_from_csv(file_path: str):
    xgb_model, transformer, static_mean, static_std = load_models()
    new_df = pd.read_csv(file_path)
    patient_data = preprocess_new_data(new_df)

    results = []
    for patient_id, (X_seq, X_static) in patient_data.items():
        proba, pred, features = predict_patient(
            xgb_model, transformer, static_mean, static_std, X_seq, X_static
        )
        results.append({
            "patient_id": patient_id,
            "prediction": int(pred),
            "probability": float(proba),
            "risk_level": "High" if proba >= 0.5 else "Low",
            "features": features
        })

    return results
