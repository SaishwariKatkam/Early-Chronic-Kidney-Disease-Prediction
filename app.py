from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pickle
import numpy as np
import traceback
import os

app = Flask(__name__)
app.secret_key = "ckd_secret_key_2024"

# ─────────────────────────────────────────────
# LOAD ALL MODEL FILES
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, 'ckd_model.joblib'))

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'label_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model_config.pkl'), 'rb') as f:
    model_config = pickle.load(f)

# Extract config
FEATURE_ORDER    = model_config['feature_order']
NUMERIC_COLS     = model_config['numeric_cols']
CATEGORICAL_COLS = model_config['categorical_cols']
CAT_MAPPINGS     = model_config['categorical_mappings']

# Median defaults (used when user leaves a field blank)
NUMERIC_DEFAULTS = {
    "age": 55.0, "bp": 80.0,  "sg": 1.02,  "al": 0.0,
    "su":  0.0,  "bgr": 121.0, "bu": 42.0,  "sc": 1.3,
    "sod": 138.0,"pot": 4.4,  "hemo": 12.65,"pcv": 40.0,
    "wc":  8000.0,"rc": 4.8
}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            processed = {}

            # --- Numeric features ---
            for col in NUMERIC_COLS:
                raw = request.form.get(col, "").strip()
                if raw == "":
                    processed[col] = NUMERIC_DEFAULTS[col]
                else:
                    processed[col] = float(raw)

            # --- Categorical features ---
            for col in CATEGORICAL_COLS:
                raw = request.form.get(col, "").strip().lower()
                if raw == "" or raw not in CAT_MAPPINGS[col]:
                    # Default to mode value (first key in mapping)
                    processed[col] = list(CAT_MAPPINGS[col].values())[0]
                else:
                    processed[col] = CAT_MAPPINGS[col][raw]

            # --- Build feature array in correct order ---
            numeric_array = np.array([[processed[col] for col in NUMERIC_COLS]])
            scaled_numeric = scaler.transform(numeric_array)[0]

            cat_array = np.array([processed[col] for col in CATEGORICAL_COLS])
            final_features = np.concatenate([scaled_numeric, cat_array]).reshape(1, -1)

            # --- Predict ---
            prediction  = model.predict(final_features)[0]
            probability = model.predict_proba(final_features)[0]
            confidence  = round(max(probability) * 100, 1)

            # prediction: 0 = ckd, 1 = notckd (from label encoder)
            le_target = label_encoders['classification']
            result_label = le_target.inverse_transform([prediction])[0]

            if result_label == 'ckd':
                result  = f"⚠️ CKD Detected"
                advice  = (
                    "The model indicates signs of Chronic Kidney Disease. "
                    "Please consult a nephrologist for proper clinical diagnosis. "
                    "This tool is for screening purposes only."
                )
                color   = "red"
            else:
                result  = f"✅ No CKD Detected"
                advice  = (
                    "The model does not indicate CKD. "
                    "Regular health checkups are still recommended. "
                    "This tool is for screening purposes only."
                )
                color   = "green"

            session['result']     = result
            session['advice']     = advice
            session['confidence'] = confidence
            session['color']      = color
            return redirect(url_for('result'))

        except Exception as e:
            print(traceback.format_exc())
            session['result']     = "❌ Error processing input"
            session['advice']     = f"Details: {str(e)}"
            session['confidence'] = 0
            session['color']      = "orange"
            return redirect(url_for('result'))

    # Pass mappings to template so dropdowns are generated dynamically
    return render_template(
        "predict.html",
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        cat_mappings=CAT_MAPPINGS,
        numeric_defaults=NUMERIC_DEFAULTS
    )


@app.route("/result")
def result():
    return render_template(
        "result.html",
        prediction_text = session.get('result',     'No result available.'),
        advice          = session.get('advice',     ''),
        confidence      = session.get('confidence', 0),
        color           = session.get('color',      'black')
    )


if __name__ == "__main__":
    app.run(debug=True)