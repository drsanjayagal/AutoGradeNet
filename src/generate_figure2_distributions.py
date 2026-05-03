#!/usr/bin/env python3
"""
generate_figure2_distributions.py
Creates kernel density plots for Surface, Deep, Structural, and Behavioural scores.
Output: ../figures/score_distributions.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Paths
    data_path = os.path.join('..', 'data_sample', 'assessment_scores.csv')
    output_path = os.path.join('..', 'figures', 'score_distributions.png')

    # Load data
    df = pd.read_csv(data_path)

    # Set style
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    # Dimensions to plot
    dimensions = ['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot KDE with fill (replacing deprecated shade)
    for dim, col in zip(dimensions, colors):
        sns.kdeplot(df[dim], label=dim.replace('Score', ''), fill=True, color=col, alpha=0.5)


    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Dimension', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure 2 saved to {output_path}")


if __name__ == "__main__":
    main()