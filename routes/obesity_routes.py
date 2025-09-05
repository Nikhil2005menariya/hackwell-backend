from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models.obesity.obesity_model import load_models, preprocess_new_data, predict_patient
import math

router = APIRouter(prefix="/predict", tags=["Obesity"])

# Load models once
xgb_model, transformer, static_mean, static_std = load_models()




def safe_float(x):
    """Convert numpy/pandas floats to Python float and replace NaN/Inf with 0."""
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    return x

@router.post("/obesity/")
async def predict_obesity(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Preprocess
    patient_data = preprocess_new_data(df)
    
    results = []
    for patient_id, (X_seq, X_static) in patient_data.items():
        proba, pred, features = predict_patient(
            xgb_model, transformer, static_mean, static_std, X_seq, X_static
        )

        # Convert all numpy types to native Python types
        features_safe = [
            {
                "name": f["name"],
                "value": float(f["value"]),
                "importance": float(f["importance"])
            }
            for f in features
        ]

        results.append({
        "patient_id": patient_id,
        "prediction": int(pred),
        "probability": safe_float(proba),
        "risk_level": "High" if proba >= 0.5 else "Low",
        "features": [
            {
                "name": f["name"],
                "value": safe_float(f["value"]),
                "importance": safe_float(f["importance"])
            } for f in features
        ]
        })

    
    return {"predictions": results}
