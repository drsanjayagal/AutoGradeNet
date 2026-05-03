#!/usr/bin/env python3
"""
generate_figure4_feature_importance.py
Trains multi-output Random Forest, computes average feature importance across dimensions,
and plots the top 20 features.
Output: ../figures/feature_importance.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Paths
    data_dir = os.path.join('..', 'data_sample')
    output_path = os.path.join('..', 'figures', 'feature_importance.png')

    # Load data files
    demographics = pd.read_csv(os.path.join(data_dir, 'student_demographics.csv'))
    assessment = pd.read_csv(os.path.join(data_dir, 'assessment_scores.csv'))
    learning = pd.read_csv(os.path.join(data_dir, 'learning_analytics.csv'))
    behavioural = pd.read_csv(os.path.join(data_dir, 'behavioural_features.csv'))
    external = pd.read_csv(os.path.join(data_dir, 'external_factors.csv'))

    # Aggregate assessment scores to student level (mean per student)
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
    X = features.drop(columns=['StudentID', 'SurfaceScore', 'DeepScore',
                               'StructuralScore', 'BehaviouralScore',
                               'ParentalInvolvement', 'InternetAccessQuality'], errors='ignore')
    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Targets: multi-dimensional scores
    y = features[['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multi-output Random Forest
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_rf)
    model.fit(X_train_scaled, y_train)

    # Average feature importance across the four output estimators
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    feature_names = X.columns

    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # top 20
    top_importances = importances[indices]
    top_features = feature_names[indices]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, align='center', color='#2c7fb8')
    plt.yticks(range(len(top_features)), top_features)
    plt.gca().invert_yaxis()
    plt.xlabel('Average Feature Importance (across 4 dimensions)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure 4 saved to {output_path}")


if __name__ == "__main__":
    main()