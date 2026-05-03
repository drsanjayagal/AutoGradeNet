#!/usr/bin/env python3
"""
generate_figure5_radar.py
Creates a radar chart summarizing per-dimension prediction performance (R², MAE)
and fairness metric (SES MAE gap). Requires evaluation results from run_evaluation.py.
Output: ../figures/summary_radar.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


def main():
    # Paths
    metrics_path = os.path.join('..', 'results', 'metrics.json')
    data_dir = os.path.join('..', 'data_sample')
    output_path = os.path.join('..', 'figures', 'summary_radar.png')

    # Load metrics (R² and MAE per dimension)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    dimensions = ['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']
    # Extract MAE and R² for each dimension
    mae_values = [metrics[dim]['MAE'] for dim in dimensions]
    r2_values = [metrics[dim]['R2'] for dim in dimensions]

    # Compute SES MAE gap per dimension
    # We need to load original data to recompute MAE per subgroup
    demographics = pd.read_csv(os.path.join(data_dir, 'student_demographics.csv'))
    assessment = pd.read_csv(os.path.join(data_dir, 'assessment_scores.csv'))
    learning = pd.read_csv(os.path.join(data_dir, 'learning_analytics.csv'))
    behavioural = pd.read_csv(os.path.join(data_dir, 'behavioural_features.csv'))
    external = pd.read_csv(os.path.join(data_dir, 'external_factors.csv'))
    labels = pd.read_csv(os.path.join(data_dir, 'ground_truth_labels.csv'))

    # Aggregate assessment scores per student
    student_scores = assessment.groupby('StudentID').agg({
        'SurfaceScore': 'mean',
        'DeepScore': 'mean',
        'StructuralScore': 'mean',
        'BehaviouralScore': 'mean'
    }).reset_index()

    # Merge features (same as in run_evaluation)
    features = demographics.merge(student_scores, on='StudentID')
    features = features.merge(learning, on='StudentID')
    features = features.merge(behavioural, on='StudentID')
    features = features.merge(external, on='StudentID')

    # For simplicity, use the same model training as run_evaluation, but we can load predictions
    # Alternatively, we recompute MAE per SES subgroup using the saved predictions?
    # Since we don't have saved per-subgroup predictions, we retrain quickly (or load model).
    # Here we retrain to be self-contained (but it's fast).
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split

    X = features.drop(columns=['StudentID', 'SurfaceScore', 'DeepScore',
                               'StructuralScore', 'BehaviouralScore',
                               'ParentalInvolvement', 'InternetAccessQuality'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    y = features[['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get test set indices and merge with SES
    test_idx = X_test.index
    test_demo = features.loc[test_idx, 'SocioeconomicStatus']

    ses_gaps = []
    for i, dim in enumerate(dimensions):
        y_true_dim = y_test.iloc[:, i].values
        y_pred_dim = y_pred[:, i]
        # Compute MAE per SES group
        ses_mae = {}
        for ses in ['Low', 'Medium', 'High']:
            mask = test_demo == ses
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true_dim[mask] - y_pred_dim[mask]))
                ses_mae[ses] = mae
        gap = ses_mae['High'] - ses_mae['Low']  # positive means high-SES has lower error (better)
        ses_gaps.append(abs(gap))  # use absolute difference for radar plot

    # Normalise metrics for radar (0 to 1 scale)
    # R² is already 0-1 (higher better) – keep as is.
    # MAE: lower better, so we invert and normalise: (max_mae - mae) / (max_mae - min_mae)
    # SES gap: lower better, normalise similarly.
    max_mae = max(mae_values)
    min_mae = min(mae_values)
    mae_norm = [(max_mae - m) / (max_mae - min_mae) for m in mae_values]

    max_gap = max(ses_gaps) if max(ses_gaps) > 0 else 1
    min_gap = min(ses_gaps) if min(ses_gaps) < max_gap else 0
    gap_norm = [(max_gap - g) / (max_gap - min_gap) if max_gap > min_gap else 1.0 for g in ses_gaps]

    # Radar chart
    # Number of variables
    N = len(dimensions)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the loop

    # Values for each metric (also close loop)
    r2_vals = r2_values + r2_values[:1]
    mae_vals = mae_norm + mae_norm[:1]
    gap_vals = gap_norm + gap_norm[:1]

    # Labels (shortened)
    labels = [d.replace('Score', '') for d in dimensions] + [dimensions[0].replace('Score', '')]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each metric as a filled polygon
    ax.plot(angles, r2_vals, 'o-', linewidth=2, label='R² (higher better)', color='#2c7fb8')
    ax.fill(angles, r2_vals, alpha=0.25, color='#2c7fb8')

    ax.plot(angles, mae_vals, 'o-', linewidth=2, label='MAE (normalised, higher better)', color='#ff7f0e')
    ax.fill(angles, mae_vals, alpha=0.25, color='#ff7f0e')

    ax.plot(angles, gap_vals, 'o-', linewidth=2, label='SES Gap (smaller gap = higher)', color='#2ca02c')
    ax.fill(angles, gap_vals, alpha=0.25, color='#2ca02c')

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure 5 saved to {output_path}")


if __name__ == "__main__":
    main()