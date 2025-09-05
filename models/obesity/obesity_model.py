import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import joblib

# -----------------------------
# FEATURES (must match training)
# -----------------------------
TS_FEATURES = [
    'bmi_day', 'waist_circumference_day', 'weight', 'systolic_bp', 
    'diastolic_bp', 'fasting_glucose', 'physical_activity_day',
    'diet_quality_day', 'symptoms', 'medication_taken', 'sleep_hours'
]

STATIC_FEATURES = [
    'age', 'sex', 'ethnicity', 'smoker', 'diabetes', 'hypertension',
    'cholesterol', 'triglycerides', 'hdl', 'ldl', 'medication_adherence',
    'physical_activity', 'alcohol_intake', 'family_history', 
    'socioeconomic_status', 'avg_sleep_hours', 'diet_quality'
]

# =====================================================
# Transformer definition
# =====================================================
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.squeeze(-1)

# =====================================================
# Load saved artifacts
# =====================================================
def load_models():
    xgb_model = joblib.load("models/obesity/xgb_hybrid_obesity_model.joblib")

    transformer = TransformerEncoder(input_dim=len(TS_FEATURES))
    transformer.load_state_dict(
        torch.load("models/obesity/transformer_encoder_obesity.pt", map_location="cpu")
    )
    transformer.eval()

    static_mean = np.load("models/obesity/scaler_static_mean_obesity.npy")
    static_std = np.load("models/obesity/scaler_static_std_obesity.npy")

    return xgb_model, transformer, static_mean, static_std

# =====================================================
# Preprocessing
# =====================================================
def preprocess_new_data(df):
    """
    Preprocess new patient data for the obesity model.
    Returns a dictionary: patient_id -> (X_seq, X_static)
    """
    # -----------------------
    # Boolean / yes-no mapping
    # -----------------------
    bool_columns = ['smoker', 'diabetes', 'hypertension', 
                    'alcohol_intake', 'family_history', 'medication_taken']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, 'YES': 1, 'NO': 0}).fillna(0)

    # -----------------------
    # Sex
    # -----------------------
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0)

    # -----------------------
    # Ethnicity and symptoms (categorical)
    # -----------------------
    if 'ethnicity' in df.columns:
        df['ethnicity'] = df['ethnicity'].astype('category').cat.codes
    if 'symptoms' in df.columns:
        df['symptoms'] = df['symptoms'].astype('category').cat.codes

    # -----------------------
    # Socioeconomic status
    # -----------------------
    if 'socioeconomic_status' in df.columns:
        status_mapping = {'low': 0, 'medium': 1, 'high': 2}
        df['socioeconomic_status'] = df['socioeconomic_status'].map(status_mapping).fillna(0)

    # -----------------------
    # Diet quality
    # -----------------------
    quality_mapping = {'poor': 0, 'fair': 1, 'good': 2, 'excellent': 3}
    if 'diet_quality' in df.columns:
        df['diet_quality'] = df['diet_quality'].map(quality_mapping).fillna(0)
    if 'diet_quality_day' in df.columns:
        df['diet_quality_day'] = df['diet_quality_day'].map(quality_mapping).fillna(0)

    # -----------------------
    # Physical activity day (categorical)
    # -----------------------
    if 'physical_activity_day' in df.columns:
        if df['physical_activity_day'].dtype == 'object':
            df['physical_activity_day'] = df['physical_activity_day'].astype('category').cat.codes

    # -----------------------
    # Convert all TS and static features to numeric
    # -----------------------
    TS_FEATURES = [
        'bmi_day', 'waist_circumference_day', 'weight', 'systolic_bp', 
        'diastolic_bp', 'fasting_glucose', 'physical_activity_day',
        'diet_quality_day', 'symptoms', 'medication_taken', 'sleep_hours'
    ]

    STATIC_FEATURES = [
        'age', 'sex', 'ethnicity', 'smoker', 'diabetes', 'hypertension',
        'cholesterol', 'triglycerides', 'hdl', 'ldl', 'medication_adherence',
        'physical_activity', 'alcohol_intake', 'family_history', 
        'socioeconomic_status', 'avg_sleep_hours', 'diet_quality'
    ]

    # Filter features that exist in df
    TS_FEATURES = [f for f in TS_FEATURES if f in df.columns]
    STATIC_FEATURES = [f for f in STATIC_FEATURES if f in df.columns]

    # Convert to numeric and fill NaNs
    df[TS_FEATURES] = df[TS_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[STATIC_FEATURES] = df[STATIC_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)

    # -----------------------
    # Build patient data dict
    # -----------------------
    patient_data = {}
    for pid, group in df.groupby('patient_id'):
        group = group.sort_values('day')
        X_seq = group[TS_FEATURES].values.astype(np.float32)
        X_static = group[STATIC_FEATURES].iloc[0].values.astype(np.float32)
        patient_data[pid] = (X_seq, X_static)

    return patient_data


# =====================================================
# Embedding extraction
# =====================================================
def get_embeddings(model, X_seq):
    model.eval()
    with torch.no_grad():
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
        x = model.input_proj(X_seq_torch)
        x = model.transformer(x)
        x = x.transpose(1, 2)
        x = model.pool(x).squeeze(-1)
        return x.numpy()

# =====================================================
# Prediction
# =====================================================
def predict_patient(xgb_model, transformer, static_mean, static_std, X_seq, X_static):
    emb = get_embeddings(transformer, X_seq)

    X_static_scaled = (X_static - static_mean) / static_std
    combined_features = np.hstack([emb.squeeze(0), X_static_scaled])

    probability = xgb_model.predict_proba(combined_features.reshape(1, -1))[0, 1]
    prediction = 1 if probability >= 0.5 else 0

    abs_sum = abs(combined_features).sum() + 1e-9
    features = [
        {
            "name": f"feature_{i+1}",
            "value": float(val),
            "importance": float(abs(val)) / abs_sum
        }
        for i, val in enumerate(combined_features)
    ]

    return probability, prediction, features
