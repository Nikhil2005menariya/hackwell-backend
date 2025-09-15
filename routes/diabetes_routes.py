from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from io import StringIO
import requests
import json
from models.diabetes.diabetes_model import load_models, preprocess_new_data, predict_patient

# Set prefix so all endpoints under this router start with /predict
router = APIRouter(prefix="/predict", tags=["Diabetes"])

# Load models once at startup
xgb_model, transformer, static_mean, static_std = load_models()

def call_gemini_api(temp_data: dict, prediction: int, disease: str):
    GEMINI_API_KEY = "my gemini key"  # 🔑 replace with env variable in production
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
# Make sure to import the json library


    # Example patient data (replace with actual data)
    temp_data = {
        "allergies": ["Peanuts", "Pollen"],
        "diet_habits": "High in processed foods, low in vegetables",
        "exercise_routine": "Sedentary, less than 1 hour/week",
        "sleep_pattern": "4-5 hours per night, irregular",
        "stress_level": "High"
    }
    
    # Corrected f-string prompt
    prompt = f'''
    You are a medical assistant AI. I will provide you with:
    1. A machine learning model's health prediction about a patient.
    2. Context data including the patient’s allergies, diet habits, exercise routines, sleep pattern, and stress levels.
    
    Your tasks:
    - Interpret the ML model’s prediction.
    - Explain in clear language why the model likely made that prediction, based on the patient’s data.
    - Suggest actionable steps the patient could take to improve or manage their health.
    - Return the output strictly in valid JSON format with the following structure:
    
    {{
      "prediction": "{{prediction}}",
      "disease": "{{disease}}",
      "explanation": "Explain why this prediction was made, referencing patient data",
      "suggestions": [
        "Suggestion 1",
        "Suggestion 2",
        "Suggestion 3"
      ]
    }}
    
    Patient Data:
    {json.dumps(temp_data, indent=2)}
    '''

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    result = response.json()

    try:
        text_output = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return json.loads(text_output)
    except Exception as e:
        return {
            "error": f"Failed to parse Gemini response: {str(e)}",
            "raw_response": result,
        }
    
# You can now print the prompt to see the final formatted string
# print(prompt)



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
        gemini_response = call_gemini_api(
            temp_data=features, prediction=int(pred), disease="Diabetes"
        )
        
        results.append({
            "patient_id": patient_id,
            "prediction": int(pred),
            "probability": float(proba),
            "risk_level": "High" if proba >= 0.5 else "Low",
            "features": features,
            "gemini_analysis": gemini_response
        })
    
    return {"predictions": results}
