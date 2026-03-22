import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('kidney_disease.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
print("\nCleaning data...")

# Drop the id column if it exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Strip whitespace from all string values
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# Replace incorrect values
df = df.replace('\t?', np.nan)
df = df.replace('?', np.nan)
df = df.replace('\t', '', regex=True)

# Clean classification column
df['classification'] = df['classification'].replace({'ckd\t': 'ckd', 'notckd': 'notckd'})

print(f"Class distribution:\n{df['classification'].value_counts()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 3. SEPARATE FEATURES AND TARGET
# ─────────────────────────────────────────────
X = df.drop('classification', axis=1)
y = df['classification']

# ─────────────────────────────────────────────
# 4. IDENTIFY COLUMN TYPES
# ─────────────────────────────────────────────
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 
                'hemo', 'pcv', 'wc', 'rc']

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

# ─────────────────────────────────────────────
# 5. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
print("\nHandling missing values...")

# Convert numeric columns to float first
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill numeric missing values with median
for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    print(f"  {col}: filled with median = {median_val:.2f}")

# Fill categorical missing values with mode
for col in categorical_cols:
    mode_val = X[col].mode()[0]
    X[col] = X[col].fillna(mode_val)
    print(f"  {col}: filled with mode = {mode_val}")

# ─────────────────────────────────────────────
# 6. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
print("\nEncoding categorical columns...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: classes = {list(le.classes_)}, encoded = {list(le.transform(le.classes_))}")

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
label_encoders['classification'] = le_target

# ─────────────────────────────────────────────
# 7. FEATURE ORDERING (IMPORTANT - must be consistent)
# ─────────────────────────────────────────────
feature_order = numeric_cols + categorical_cols
X = X[feature_order]
print(f"\nFinal feature order: {feature_order}")

# ─────────────────────────────────────────────
# 8. SCALE NUMERIC FEATURES
# ─────────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ─────────────────────────────────────────────
# 9. TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded  # Important for imbalanced data
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 10. TRAIN MODEL
# ─────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handles class imbalance
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 11. EVALUATE MODEL PROPERLY
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, 
      target_names=le_target.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted CKD  Predicted NotCKD")
print(f"Actual CKD          {cm[0][0]}              {cm[0][1]}")
print(f"Actual NotCKD       {cm[1][0]}              {cm[1][1]}")

# Feature importance
print("\nTop 10 Most Important Features:")
importances = model.feature_importances_
feature_names = feature_order
feat_imp = sorted(zip(feature_names, importances), 
                  key=lambda x: x[1], reverse=True)[:10]
for feat, imp in feat_imp:
    print(f"  {feat}: {imp:.4f}")

# ─────────────────────────────────────────────
# 12. SAVE ALL FILES CORRECTLY
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("SAVING MODEL FILES")
print("="*50)

# Save model
joblib.dump(model, 'ckd_model.joblib')
print("✅ Saved: ckd_model.joblib")

# Save label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✅ Saved: label_encoders.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: scaler.pkl")

# Save feature order and column info for Flask app
model_config = {
    'feature_order': feature_order,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'categorical_mappings': {}
}

# Save human-readable mappings for Flask dropdowns
for col in categorical_cols:
    le = label_encoders[col]
    model_config['categorical_mappings'][col] = {
        cls: int(le.transform([cls])[0]) 
        for cls in le.classes_
    }

with open('model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)
print("✅ Saved: model_config.pkl")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print("\nModel config summary:")
for col, mapping in model_config['categorical_mappings'].items():
    print(f"  {col}: {mapping}")

print("\nYou can now run: python app.py")