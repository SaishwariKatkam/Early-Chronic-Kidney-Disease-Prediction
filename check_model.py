import pickle
import joblib

# Check label encoders
print("=" * 40)
print("LABEL ENCODERS:")
print("=" * 40)
with open('label_encoders.pkl', 'rb') as f:
    le = pickle.load(f)
print("Type:", type(le))
print("Contents:", le)
print()

# Check scaler
print("=" * 40)
print("SCALER:")
print("=" * 40)
with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)
print("Type:", type(sc))
print("Feature names:", sc.feature_names_in_ if hasattr(sc, 'feature_names_in_') else "Not available")
print("Mean values:", sc.mean_ if hasattr(sc, 'mean_') else "Not available")
print()

# Check model
print("=" * 40)
print("MODEL:")
print("=" * 40)
model = joblib.load('ckd_model.joblib')
print("Type:", type(model))
print("Features expected:", model.n_features_in_ if hasattr(model, 'n_features_in_') else "Not available")
print()

# Quick prediction test
import numpy as np
print("=" * 40)
print("QUICK PREDICTION TEST:")
print("=" * 40)
try:
    # Using 24 features - all zeros just to test shape
    test = np.zeros((1, model.n_features_in_))
    pred = model.predict(test)
    print("Model accepted input shape correctly")
    print("Test prediction:", pred)
except Exception as e:
    print("Error:", e)