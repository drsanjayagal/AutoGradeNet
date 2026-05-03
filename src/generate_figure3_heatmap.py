#!/usr/bin/env python3
"""
generate_figure3_heatmap.py
Creates a correlation heatmap of Surface, Deep, Structural, Behavioural scores and Final Grade.
Output: ../figures/correlation_heatmap.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Paths
    assess_path = os.path.join('..', 'data_sample', 'assessment_scores.csv')
    labels_path = os.path.join('..', 'data_sample', 'ground_truth_labels.csv')
    output_path = os.path.join('..', 'figures', 'correlation_heatmap.png')

    # Load data
    assess = pd.read_csv(assess_path)
    labels = pd.read_csv(labels_path)

    # Aggregate assessment scores per student (mean)
    dim_scores = assess.groupby('StudentID')[
        ['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']].mean().reset_index()

    # Merge with final grade
    merged = dim_scores.merge(labels[['StudentID', 'FinalGrade']], on='StudentID', how='inner')

    # Compute correlation matrix
    corr = merged[['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore', 'FinalGrade']].corr()

    # Rename columns/rows for nicer labels
    corr.columns = ['Surface', 'Deep', 'Structural', 'Behavioural', 'Final']
    corr.index = ['Surface', 'Deep', 'Structural', 'Behavioural', 'Final']

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt='.2f', square=True, linewidths=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure 3 saved to {output_path}")


if __name__ == "__main__":
    main()