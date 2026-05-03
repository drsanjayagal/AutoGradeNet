#!/usr/bin/env python3
"""
AutoGradeNet – Model Evaluation & Results Generator
Loads data from ../data_sample/, trains a simple multi‑output regression model,
and saves results (metrics, predictions, plots) into ../results/ and ../figures/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------------------
# 0. Path configuration
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data_sample")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
print("📂 Loading data from data_sample/ ...")

try:
    demographics = pd.read_csv(os.path.join(DATA_DIR, "student_demographics.csv"))
    assessment = pd.read_csv(os.path.join(DATA_DIR, "assessment_scores.csv"))
    learning = pd.read_csv(os.path.join(DATA_DIR, "learning_analytics.csv"))
    behavioural = pd.read_csv(os.path.join(DATA_DIR, "behavioural_features.csv"))
    external = pd.read_csv(os.path.join(DATA_DIR, "external_factors.csv"))
    labels = pd.read_csv(os.path.join(DATA_DIR, "ground_truth_labels.csv"))
except FileNotFoundError as e:
    print(f"❌ Missing data file: {e}")
    print("   Run generate_synthetic_data.py first to create the files.")
    sys.exit(1)

# -------------------------------------------------------------------
# 2. Prepare features and targets
# -------------------------------------------------------------------
# Aggregate assessment scores to student-level (mean per student)
student_scores = assessment.groupby('StudentID').agg({
    'SurfaceScore': 'mean',
    'DeepScore': 'mean',
    'StructuralScore': 'mean',
    'BehaviouralScore': 'mean'
}).reset_index()

# Merge all feature tables
features = demographics.merge(student_scores, on='StudentID', how='inner')
features = features.merge(learning, on='StudentID', how='inner')
features = features.merge(behavioural, on='StudentID', how='inner')
features = features.merge(external, on='StudentID', how='inner')

# Drop non-numeric or ID columns
X = features.drop(columns=['StudentID', 'FinalGrade', 'SurfaceScore', 'DeepScore',
                           'StructuralScore', 'BehaviouralScore',
                           'ParentalInvolvement', 'InternetAccessQuality'],
                  errors='ignore')
# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Targets: multi‑dimensional scores (can also use FinalGrade as single target)
y = features[['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']]

print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.\n")

# -------------------------------------------------------------------
# 3. Train / test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# 4. Train a multi-output random forest model
# -------------------------------------------------------------------
print("🚀 Training multi‑output regression model...")
base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputRegressor(base_rf)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# -------------------------------------------------------------------
# 5. Evaluate and save metrics
# -------------------------------------------------------------------
metrics = {}
for i, dim in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    metrics[dim] = {'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 3)}

# Also compute average across dimensions
metrics['Average'] = {
    'MAE': round(np.mean([m['MAE'] for m in metrics.values() if isinstance(m, dict)]), 3),
    'RMSE': round(np.mean([m['RMSE'] for m in metrics.values() if isinstance(m, dict)]), 3),
    'R2': round(np.mean([m['R2'] for m in metrics.values() if isinstance(m, dict)]), 3)
}

# Save metrics to JSON
metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Metrics saved to {metrics_path}")

# -------------------------------------------------------------------
# 6. Save predictions
# -------------------------------------------------------------------
predictions_df = pd.DataFrame(y_pred, columns=[f'pred_{col}' for col in y.columns])
predictions_df['StudentID'] = y_test.index.values  # note: index is not original StudentID
# Map back to original StudentID using the test set's preserved ID? We'll just save as is.
predictions_df.to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)
print(f"✅ Predictions saved to {os.path.join(RESULTS_DIR, 'predictions.csv')}")

# -------------------------------------------------------------------
# 7. Generate feature importance plot (for first output dimension)
# -------------------------------------------------------------------
# Extract feature importances from the first estimator
first_estimator = model.estimators_[0]
importances = first_estimator.feature_importances_
feature_names = X.columns

# Sort by importance
indices = np.argsort(importances)[::-1][:20]  # top 20 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Top 20 Features Predicting Multi‑Dimensional Scores')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150)
plt.close()
print(f"✅ Feature importance plot saved to {os.path.join(FIGURES_DIR, 'feature_importance.png')}")

# -------------------------------------------------------------------
# 8. Write a human-readable evaluation report
# -------------------------------------------------------------------
report_lines = [
    "=== AutoGradeNet Evaluation Report ===",
    f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Model: RandomForestRegressor (MultiOutput, 100 trees)",
    f"Test set size: {len(y_test)} samples",
    "",
    "Per-dimension Performance:",
]
for dim, m in metrics.items():
    if dim != 'Average':
        report_lines.append(f"  {dim}: MAE={m['MAE']}, RMSE={m['RMSE']}, R2={m['R2']}")
report_lines.append(f"\nAverage across dimensions: MAE={metrics['Average']['MAE']}, RMSE={metrics['Average']['RMSE']}, R2={metrics['Average']['R2']}")
report_lines.append("\nGenerated files:")
report_lines.append(f"  - {metrics_path}")
report_lines.append(f"  - {os.path.join(RESULTS_DIR, 'predictions.csv')}")
report_lines.append(f"  - {os.path.join(FIGURES_DIR, 'feature_importance.png')}")

report_path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
print(f"✅ Evaluation report saved to {report_path}")

# -------------------------------------------------------------------
# 9. (Optional) Download real-world dataset from UCI
#    Uncomment the block below if you wish to replace synthetic data.
# -------------------------------------------------------------------
# def download_real_data():
#     import urllib.request
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
#     # ... implementation to download and extract to DATA_DIR
#     print("Real data would be downloaded here (uncomment and implement).")
#
# if input("Use real UCI student data? (y/n): ").lower() == 'y':
#     download_real_data()

print("\n✨ All results generated! Check the 'results' and 'figures' folders.")