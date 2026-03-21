from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import traceback
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Use a strong secret key in production

# Safe path to the model relative to app location
model_path = os.path.join(os.path.dirname(__file__), "ckd_model.joblib")
model = joblib.load(model_path)

# List of expected input field names
input_fields = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
    "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect and validate input data
            values = []
            for field in input_fields:
                value = request.form.get(field)
                if value is None or value.strip() == "":
                    raise ValueError(f"Missing or empty value for field: {field}")
                values.append(float(value))

            # Prepare input array
            features = np.array([values])
            prediction = model.predict(features)[0]
            result = "CKD Detected" if prediction == 1 else "No CKD"

            session["prediction_result"] = result
            return redirect(url_for("result"))

        except Exception as e:
            print(traceback.format_exc())  # Debugging info in terminal
            session["prediction_result"] = f"Invalid input. Error: {str(e)}"
            return redirect(url_for("result"))

    return render_template("predict.html")

@app.route("/result")
def result():
    result = session.get("prediction_result", "No result available.")
    return render_template("result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)