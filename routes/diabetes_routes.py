from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
from models.diabetes.diabetes_model import load_models, preprocess_new_data, predict_patient

# Set prefix so all endpoints under this router start with /predict
router = APIRouter(prefix="/predict", tags=["Diabetes"])

# Load models once at startup
xgb_model, transformer, static_mean, static_std = load_models()

@router.post("/diabetes/")
async def predict_diabetes(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    
    patient_data = preprocess_new_data(df)
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
            "features": features   # ✅ send to frontend
        })
    
    return {"predictions": results}