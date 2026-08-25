import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv("pcos_adolescent_train_dataset.csv")

# Adolescent-specific adjustment: irregular periods within 2 years of menarche
# are clinically expected and shouldn't count as a strong signal.
df["Irregular_Periods_Adj"] = df["Irregular_Periods"] * (df["Years_Since_Menarche"] >= 2).astype(int)

features = [
    "Age", "Years_Since_Menarche", "BMI", "Acne",
    "Irregular_Periods_Adj", "Facial_Hair_Growth", "Weight_Gain",
    "Bloating", "Hair_Thinning", "Fatigue",
]
X = df[features]
y = df["PCOS_Risk_Indicator"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

print("=== Feature weights (higher = pushes toward 'some indicators') ===")
for name, coef in sorted(zip(features, model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {name:25s} {coef:+.3f}")

print("\n=== Threshold comparison ===")
for t in [0.5, 0.6, 0.65, 0.7]:
    preds = (probs >= t).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"threshold={t:.2f}  precision(1)={report['1']['precision']:.2f}  "
          f"recall(1)={report['1']['recall']:.2f}  false_positives_avoided_vs_0.5=n/a")

print(f"\nROC-AUC: {roc_auc_score(y_test, probs):.3f}")

THRESHOLD = 0.65
final_preds = (probs >= THRESHOLD).astype(int)
print(f"\n=== Final report @ threshold={THRESHOLD} ===")
print(classification_report(y_test, final_preds, zero_division=0))

joblib.dump({"model": model, "features": features, "threshold": THRESHOLD}, "pcos_model.joblib")
print("\nSaved pcos_model.joblib")
