import numpy as np
import torch
import torch.nn as nn
import joblib

# -----------------------------
# FEATURES (must match training)
# -----------------------------
# Time-series features used during training (17)
TS_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk'
]

# Static features used during training (4)
STATIC_FEATURES = ['Sex', 'Age', 'Education', 'Income']

# =====================================================
# Transformer definition (same architecture used in training)
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
        # classifier head is not used for embedding extraction but keep signature
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.squeeze(-1)

# =====================================================
# Load saved artifacts (paths must match where you put files)
# =====================================================
def load_models():
    """
    Loads xgboost model, transformer weights and static scalers.
    """
    xgb_model = joblib.load("models/diabetes/xgb_hybrid_model.joblib")

    transformer = TransformerEncoder(input_dim=len(TS_FEATURES))
    transformer.load_state_dict(torch.load("models/diabetes/transformer_encoder.pt", map_location="cpu"))
    transformer.eval()

    static_mean = np.load("models/diabetes/scaler_static_mean.npy")
    static_std = np.load("models/diabetes/scaler_static_std.npy")

    return xgb_model, transformer, static_mean, static_std

# =====================================================
# Preprocessing utility (used by route)
# =====================================================
def preprocess_new_data(df):
    """
    Given a pandas DataFrame loaded from CSV, produce a dict:
      { patient_id: (X_seq (timesteps x ts_features), X_static (len STATIC_FEATURES)) }
    This function mirrors the preprocessing used during training (Sex encoding, order of features).
    """
    # Ensure Sex encoded exactly like training
    if 'Sex' in df.columns:
        df['Sex'] = (df['Sex'].astype(str).str.lower() == 'male').astype(int)

    patient_data = {}

    for pid, group in df.groupby('patient_id'):
        group = group.sort_values('day')

        # Verify time-series columns exist
        missing_ts = [c for c in TS_FEATURES if c not in group.columns]
        if missing_ts:
            raise ValueError(f"Missing time-series columns for patient {pid}: {missing_ts}")

        # Extract sequential features (timesteps x features)
        X_seq = group[TS_FEATURES].values.astype(np.float32)  # shape (timesteps, ts_features)

        # Verify static columns exist
        missing_static = [c for c in STATIC_FEATURES if c not in group.columns]
        if missing_static:
            raise ValueError(f"Missing static columns for patient {pid}: {missing_static}")

        # Extract static features (take first row)
        X_static_raw = group[STATIC_FEATURES].iloc[0]

        # Convert static features to numeric safely (Sex already encoded above)
        X_static = []
        for col in STATIC_FEATURES:
            val = X_static_raw[col]
            try:
                X_static.append(float(val))
            except Exception:
                # fallback for common categorical strings (if any)
                if col == 'Sex':
                    X_static.append(1.0 if str(val).lower() == 'male' else 0.0)
                else:
                    # fallback numeric extraction or 0.0
                    try:
                        # extract digits and decimal if present
                        filtered = ''.join(ch for ch in str(val) if (ch.isdigit() or ch in '.-'))
                        X_static.append(float(filtered) if filtered not in ('', '.', '-', '-.') else 0.0)
                    except Exception:
                        X_static.append(0.0)

        X_static = np.array(X_static, dtype=np.float32)  # shape (len(STATIC_FEATURES),)

        patient_data[pid] = (X_seq, X_static)

    return patient_data

# =====================================================
# Embedding extraction
# =====================================================
def get_embeddings(model, X_seq):
    """
    Returns numpy embedding for a single patient X_seq (timesteps x ts_features).
    Output shape: (1, d_model)
    """
    model.eval()
    with torch.no_grad():
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # (1, timesteps, ts_features)
        x = model.input_proj(X_seq_torch)
        x = model.transformer(x)
        x = x.transpose(1, 2)
        x = model.pool(x).squeeze(-1)  # (1, d_model)
        return x.numpy()

# =====================================================
# Prediction & feature extraction
# =====================================================
def predict_patient(xgb_model, transformer, static_mean, static_std, X_seq, X_static):
    """Make prediction for a single patient and return feature values."""
    # Get embeddings from transformer
    emb = get_embeddings(transformer, X_seq)

    # Scale static features
    X_static_scaled = (X_static - static_mean) / static_std

    # Combine embeddings + static
    combined_features = np.hstack([emb.squeeze(0), X_static_scaled])

    # Make prediction
    probability = xgb_model.predict_proba(combined_features.reshape(1, -1))[0, 1]
    prediction = 1 if probability >= 0.5 else 0

    # Build features list with normalized importance (just a simple proxy for now)
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
