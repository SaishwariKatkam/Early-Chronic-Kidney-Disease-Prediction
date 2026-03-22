# Early Prediction of Chronic Kidney Disease 🩺
### End-to-End Machine Learning Pipeline with Flask Deployment

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.1-000000?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.1-F7931E?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.3.1-150458?style=flat&logo=pandas)
![Accuracy](https://img.shields.io/badge/Accuracy-97.5%25-brightgreen?style=flat)

---

## 🔍 What is this project?

An end-to-end machine learning web application that predicts whether a patient has **Chronic Kidney Disease (CKD)** based on 24 clinical parameters. The model is trained on real patient data, properly preprocessed, and deployed via a Flask web interface with confidence scoring.

---

## ✨ Key Highlights

- **97.5% accuracy** on held-out test set
- **100% CKD recall** — zero false negatives (no CKD case missed)
- Handles **missing clinical data** gracefully using median/mode imputation
- Shows **confidence percentage** with every prediction
- Includes **medical disclaimer** for responsible AI usage
- Clean, responsive web UI with navigation

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset

| Property | Value |
|---|---|
| Total Patients | 400 |
| CKD Cases | 250 (62.5%) |
| Non-CKD Cases | 150 (37.5%) |
| Features | 24 clinical parameters |
| Missing Values | Handled via median/mode imputation |

### Features Used

| Type | Features |
|---|---|
| Numeric | Age, Blood Pressure, Specific Gravity, Albumin, Sugar, Blood Glucose, Blood Urea, Serum Creatinine, Sodium, Potassium, Hemoglobin, Packed Cell Volume, WBC Count, RBC Count |
| Categorical | Red Blood Cells, Pus Cell, Pus Cell Clumps, Bacteria, Hypertension, Diabetes Mellitus, Coronary Artery Disease, Appetite, Pedal Edema, Anemia |

---

## 🧠 ML Pipeline

```
Raw CSV Data
     ↓
Data Cleaning
(strip whitespace, fix inconsistent values)
     ↓
Missing Value Imputation
(median for numeric, mode for categorical)
     ↓
Label Encoding
(categorical → 0/1)
     ↓
Standard Scaling
(numeric features normalized)
     ↓
Train/Test Split (80/20, stratified)
     ↓
Random Forest Classifier
(class_weight='balanced' for imbalance)
     ↓
Evaluation + Model Saving
(joblib + pickle)
     ↓
Flask Web App Deployment
```

---

## 📈 Model Performance

| Metric | CKD | Not CKD |
|---|---|---|
| Precision | 96% | 100% |
| Recall | 100% | 93% |
| F1-Score | 98% | 97% |
| **Overall Accuracy** | **97.5%** | |

### Confusion Matrix
```
                 Predicted CKD   Predicted Not CKD
Actual CKD            50               0        ← Zero missed!
Actual Not CKD         2              28
```

### Top Predictive Features
```
1. Hemoglobin          23.47%
2. Packed Cell Volume  15.11%
3. Specific Gravity    14.41%
4. Serum Creatinine    12.37%
5. Red Blood Cell Count 9.32%
```
These align with real clinical indicators of kidney disease — validating that the model learned genuine patterns, not noise.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| ML | Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Model Serialization | Joblib, Pickle |
| Frontend | HTML, CSS |

---

## 📁 Project Structure

```
Early Chronic Kidney Disease Prediction/
│
├── templates/
│   ├── home.html          # Landing page
│   ├── predict.html       # Input form (24 parameters)
│   └── result.html        # Prediction result with confidence
│
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── kidney_disease.csv     # Dataset (UCI)
│
├── ckd_model.joblib       # Trained Random Forest model
├── scaler.pkl             # StandardScaler for numeric features
├── label_encoders.pkl     # LabelEncoders for categorical features
├── model_config.pkl       # Feature order + categorical mappings
│
└── requirements.txt       # Python dependencies
```

---

## ⚡ Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/[yourusername]/ckd-prediction.git
cd ckd-prediction
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model (generates all .pkl and .joblib files)**
```bash
python train_model.py
```

**5. Run the Flask app**
```bash
python app.py
```

**6. Open in browser**
```
http://127.0.0.1:5000
```

---

## 🖥️ How to Use

1. Navigate to the **Predict** page
2. Enter patient clinical parameters
3. Leave any unknown fields blank (defaults to dataset median)
4. Click **Predict**
5. View result with confidence percentage and medical advice

---

## ⚠️ Medical Disclaimer

This application is a **screening tool only** and is not intended for clinical diagnosis. All predictions should be verified by a qualified medical professional. The model has a 2.5% error rate and should never replace professional medical advice.

---

## 🔮 Future Improvements
- [ ] Add more ML models (XGBoost, SVM) for comparison
- [ ] Deploy on AWS (EC2 + S3)
- [ ] Add patient history tracking with database
- [ ] Improve UI with React frontend
- [ ] Add SHAP explanations for individual predictions

---

## 👩‍💻 Author

**Saishwari Venkatesh Katkam**
MIT Academy of Engineering, Pune | B.Tech Computer Engineering 2027
[LinkedIn](https://www.linkedin.com/in/saishwari-katkam-aa5597324/) | [GitHub](https://github.com/SaishwariKatkam)

---

## 📄 License
MIT License
