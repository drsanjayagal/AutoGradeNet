#!/usr/bin/env python3
"""
generate_framework_diagram.py
Creates a high-resolution architectural diagram of AutoGradeNet.
Output: ../figures/framework_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def main():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors
    color_data = '#4A90E2'  # blue
    color_ml = '#50E3C2'  # teal
    color_stats = '#F5A623'  # orange
    color_fair = '#D0021B'  # red
    color_xai = '#9013FE'  # purple

    # Define boxes: (x_center, y_center, width, height, text, color)
    boxes = [
        (5, 7.0, 8.0, 0.8, "Raw Student Data\n(Demographics, Assessments, Logs, Behaviour, External)", color_data),
        (5, 6.0, 4.0, 0.8, "Data Engine\n(cleaning, aggregation, encoding)", color_data),
        (5, 4.8, 4.0, 0.8, "ML Engine (Multi‑Output Random Forest)\nPredicts Surface, Deep, Structural, Behavioural",
         color_ml),
        (2.5, 3.2, 2.5, 0.8, "Statistical Engine\n(descriptives, correlation, ANOVA)", color_stats),
        (5.0, 3.2, 2.5, 0.8, "Fairness Engine\n(gender, SES, language subgroups)", color_fair),
        (7.5, 3.2, 2.5, 0.8, "Explainability Layer\n(feature importance, SHAP)", color_xai),
        (5, 1.8, 8.0, 0.8, "Reports & Visualisations\n(metrics.json, predictions.csv, figures/)", color_stats),
    ]

    # Draw boxes
    for x, y, w, h, text, col in boxes:
        rect = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                      boxstyle=patches.BoxStyle("Round", pad=0.1),
                                      facecolor=col, edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Draw arrows
    arrow_kw = dict(arrowstyle='->', color='black', lw=2, connectionstyle='arc3,rad=0')
    # Data flow
    ax.annotate("", xy=(5, 6.4), xytext=(5, 6.6), arrowprops=arrow_kw)
    ax.annotate("", xy=(5, 5.2), xytext=(5, 5.6), arrowprops=arrow_kw)
    # Split to three engines
    ax.annotate("", xy=(2.5, 3.6), xytext=(4, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.0, 3.6), xytext=(5, 4.4), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.5, 3.6), xytext=(6, 4.4), arrowprops=arrow_kw)
    # Merge back to reports
    ax.annotate("", xy=(5, 2.2), xytext=(2.5, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5, 2.2), xytext=(5, 2.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(5, 2.2), xytext=(7.5, 2.8), arrowprops=arrow_kw)


    plt.tight_layout()
    plt.savefig('../figures/framework_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Framework diagram saved to ../figures/framework_diagram.png")


if __name__ == "__main__":
    main()