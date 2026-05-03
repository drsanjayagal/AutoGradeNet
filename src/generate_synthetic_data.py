#!/usr/bin/env python3
"""
AutoGradeNet – Synthetic Data Generator
Outputs CSV files into ../data_sample/ and plots into ../figures/
Run this script from the project root or from the src folder.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------------------------------------------
# 0. Path configuration (script is inside /src, outputs go up one level)
# -------------------------------------------------------------------
# Get the directory where this script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is one level up from src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define output directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data_sample")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
np.random.seed(42)

NUM_STUDENTS = 5000
NUM_ASSESSMENTS = 4  # Assessments per student per subject
NUM_SUBJECTS = 4
NUM_WEEKS = 16

SUBJECTS = ['Mathematics', 'Physics', 'English_Literature', 'History']
ASSESSMENT_TYPES = ['Quiz', 'Midterm', 'Assignment', 'Final']
GENDER = ['Male', 'Female', 'Non-binary']
SOCIO_ECON = ['Low', 'Medium', 'High']
EDUC_BACKGROUND = ['Below_Secondary', 'Secondary', 'Bachelors', 'Masters_Plus']
LANGUAGE_BACKGROUND = ['Native', 'Bilingual', 'ESL']
DIMENSIONS = ['Surface', 'Deep', 'Structural', 'Behavioural']


# -------------------------------------------------------------------
# 2. Generate Student Demographics
# -------------------------------------------------------------------
def generate_demographics(n):
    demographics = pd.DataFrame({
        'StudentID': range(1, n + 1),
        'Age': np.random.randint(18, 31, n),
        'Gender': np.random.choice(GENDER, n, p=[0.48, 0.48, 0.04]),
        'SocioeconomicStatus': np.random.choice(SOCIO_ECON, n, p=[0.25, 0.55, 0.20]),
        'ParentalEducation': np.random.choice(EDUC_BACKGROUND, n, p=[0.15, 0.35, 0.35, 0.15]),
        'LanguageBackground': np.random.choice(LANGUAGE_BACKGROUND, n, p=[0.65, 0.20, 0.15]),
        'PriorGPA': np.random.normal(2.8, 0.6, n).clip(0, 4)
    })
    return demographics


# -------------------------------------------------------------------
# 3. Generate Assessment Scores (multi-dimensional)
# -------------------------------------------------------------------
def generate_assessment_scores(demographics):
    records = []
    for _, student in demographics.iterrows():
        ability_base = student['PriorGPA'] * 12 + 40
        if student['SocioeconomicStatus'] == 'Low':
            ability_base -= 5
        elif student['SocioeconomicStatus'] == 'High':
            ability_base += 5

        for subject in SUBJECTS:
            if subject == 'Mathematics':
                subject_bias = 2 if student['Gender'] == 'Male' else -2
            elif subject == 'English_Literature':
                subject_bias = 3 if student['LanguageBackground'] == 'Native' else -3
            else:
                subject_bias = 0

            for assess_type in ASSESSMENT_TYPES:
                if assess_type == 'Quiz':
                    difficulty = np.random.normal(0, 5)
                elif assess_type == 'Midterm':
                    difficulty = np.random.normal(-3, 7)
                elif assess_type == 'Assignment':
                    difficulty = np.random.normal(2, 6)
                else:
                    difficulty = np.random.normal(-1, 8)

                raw_score = ability_base + subject_bias + difficulty + np.random.normal(0, 8)
                raw_score = max(0, min(100, raw_score))

                mean_vec = [raw_score * 0.25] * 4
                cov_mat = np.array([[100, 60, 50, 40],
                                    [60, 100, 55, 45],
                                    [50, 55, 100, 50],
                                    [40, 45, 50, 100]])
                dim_scores = np.random.multivariate_normal(mean_vec, cov_mat, 1)[0]
                dim_scores = np.clip(dim_scores, 0, 100)

                records.append({
                    'StudentID': student['StudentID'],
                    'Subject': subject,
                    'AssessmentType': assess_type,
                    'Week': np.random.randint(1, NUM_WEEKS + 1),
                    'TotalScore': raw_score,
                    'SurfaceScore': dim_scores[0],
                    'DeepScore': dim_scores[1],
                    'StructuralScore': dim_scores[2],
                    'BehaviouralScore': dim_scores[3]
                })
    return pd.DataFrame(records)


# -------------------------------------------------------------------
# 4. Generate Learning Analytics
# -------------------------------------------------------------------
def generate_learning_analytics(demographics, assessment_scores):
    n_assessments = assessment_scores.groupby('StudentID').size().reset_index(name='NAssessments')
    n_assessments['NAssessments'] = n_assessments['NAssessments'].astype(int)
    n = len(demographics)
    analytics = pd.DataFrame({
        'StudentID': range(1, n + 1),
        'TotalLogins': np.random.poisson(40, n).clip(5, 80),
        'ForumPosts': np.random.poisson(8, n).clip(0, 45),
        'AssignmentViews': np.random.poisson(18, n).clip(2, 65),
        'QuizAttempts': np.random.poisson(6, n).clip(1, 20),
        'ResourceAccessCount': np.random.poisson(25, n).clip(5, 85)
    })
    analytics = analytics.merge(n_assessments, on='StudentID', how='left')
    analytics['NAssessments'] = analytics['NAssessments'].fillna(0).astype(int)
    return analytics


# -------------------------------------------------------------------
# 5. Generate Behavioural Features
# -------------------------------------------------------------------
def generate_behavioural_features(demographics):
    n = len(demographics)
    behavioural = pd.DataFrame({
        'StudentID': range(1, n + 1),
        'LateSubmissions': np.random.poisson(2, n).clip(0, 12),
        'SkippedQuestions': np.random.poisson(1.5, n).clip(0, 8),
        'HelpRequests': np.random.poisson(5, n).clip(0, 20),
        'AttendanceRate': np.random.beta(7, 3, n) * 100,
        'AvgResponseTime': np.random.exponential(120, n).clip(10, 600)
    })
    return behavioural


# -------------------------------------------------------------------
# 6. Generate Temporal Engagement Data
# -------------------------------------------------------------------
def generate_temporal_engagement(demographics):
    records = []
    for _, student in demographics.iterrows():
        for week in range(1, NUM_WEEKS + 1):
            base = 10
            decay = -0.1 * week
            engagement = base + decay + np.random.normal(0, 1.5)
            engagement = max(0, min(20, engagement))
            records.append({
                'StudentID': student['StudentID'],
                'Week': week,
                'EngagementScore': engagement
            })
    return pd.DataFrame(records)


# -------------------------------------------------------------------
# 7. Generate External Factors
# -------------------------------------------------------------------
def generate_external_factors(demographics):
    n = len(demographics)
    external = pd.DataFrame({
        'StudentID': range(1, n + 1),
        'SleepHours': np.random.normal(7, 1.2, n).clip(3, 12),
        'StressLevel': np.random.normal(5, 1.5, n).clip(1, 10),
        'ParentalInvolvement': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.2, 0.6, 0.2]),
        'InternetAccessQuality': np.random.choice(['Poor', 'Average', 'Good'], n, p=[0.1, 0.5, 0.4])
    })
    return external


# -------------------------------------------------------------------
# 8. Generate Ground-Truth Labels
# -------------------------------------------------------------------
def generate_ground_truth(assessment_scores, learning_analytics, behavioural_features, external_factors):
    agg_assess = assessment_scores.groupby('StudentID').agg({
        'SurfaceScore': 'mean',
        'DeepScore': 'mean',
        'StructuralScore': 'mean',
        'BehaviouralScore': 'mean'
    }).reset_index()

    agg_learn = learning_analytics.drop(columns=['NAssessments']).groupby('StudentID').mean().reset_index()
    merged = agg_assess.merge(agg_learn, on='StudentID', how='left')
    merged = merged.merge(behavioural_features, on='StudentID', how='left')
    merged = merged.merge(external_factors.drop(columns=['InternetAccessQuality']), on='StudentID', how='left')

    final_score = (0.35 * merged['SurfaceScore'] +
                   0.30 * merged['DeepScore'] +
                   0.20 * merged['StructuralScore'] +
                   0.15 * merged['BehaviouralScore'])
    final_score -= 0.5 * merged['LateSubmissions']
    final_score -= 0.5 * merged['SkippedQuestions']
    final_score += 0.1 * merged['TotalLogins']

    final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 100
    final_score = final_score.clip(0, 100)

    ground_truth = pd.DataFrame({'StudentID': merged['StudentID'], 'FinalGrade': final_score})
    ground_truth = ground_truth.merge(agg_assess, on='StudentID', how='left')
    return ground_truth


# -------------------------------------------------------------------
# 9. Plot and Save Figures
# -------------------------------------------------------------------
def generate_figures(assessment_scores, ground_truth):
    # Figure 1: Distribution of multi-dimensional scores
    plt.figure(figsize=(10, 6))
    for dim in DIMENSIONS:
        sns.kdeplot(assessment_scores[f'{dim}Score'], label=dim, shade=True)
    plt.title('Distribution of Multi‑Dimensional Assessment Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'score_distributions.png'), dpi=150)
    plt.close()

    # Figure 2: Correlation heatmap of ground truth dimensions
    corr_cols = ['SurfaceScore', 'DeepScore', 'StructuralScore', 'BehaviouralScore', 'FinalGrade']
    corr = ground_truth[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation between Multi‑Dimensional Scores and Final Grade')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    print(f"✅ Figures saved to {FIGURES_DIR}")


# -------------------------------------------------------------------
# 10. Main Execution
# -------------------------------------------------------------------
def main():
    print("🚀 Generating synthetic data for AutoGradeNet...")
    print(f"   Data will be saved to: {DATA_DIR}")
    print(f"   Figures will be saved to: {FIGURES_DIR}\n")

    # Generate all dataframes
    demo = generate_demographics(NUM_STUDENTS)
    assmt = generate_assessment_scores(demo)
    learn = generate_learning_analytics(demo, assmt)
    behav = generate_behavioural_features(demo)
    temporal = generate_temporal_engagement(demo)
    external = generate_external_factors(demo)
    labels = generate_ground_truth(assmt, learn, behav, external)

    # Save CSV files
    demo.to_csv(os.path.join(DATA_DIR, 'student_demographics.csv'), index=False)
    assmt.to_csv(os.path.join(DATA_DIR, 'assessment_scores.csv'), index=False)
    learn.to_csv(os.path.join(DATA_DIR, 'learning_analytics.csv'), index=False)
    behav.to_csv(os.path.join(DATA_DIR, 'behavioural_features.csv'), index=False)
    temporal.to_csv(os.path.join(DATA_DIR, 'temporal_engagement.csv'), index=False)
    external.to_csv(os.path.join(DATA_DIR, 'external_factors.csv'), index=False)
    labels.to_csv(os.path.join(DATA_DIR, 'ground_truth_labels.csv'), index=False)

    print("✅ CSV files saved to data_sample/")

    # Generate and save figures
    generate_figures(assmt, labels)

    print("\n📊 Summary Statistics:")
    print(f"   Number of students: {NUM_STUDENTS}")
    print(f"   Average Final Grade: {labels['FinalGrade'].mean():.2f}")
    print(f"   Average Surface Score: {assmt['SurfaceScore'].mean():.2f}")
    print(f"   Average Deep Score: {assmt['DeepScore'].mean():.2f}")
    print(f"   Average Structural Score: {assmt['StructuralScore'].mean():.2f}")
    print(f"   Average Behavioural Score: {assmt['BehaviouralScore'].mean():.2f}")
    print("\n✨ All data and figures generated successfully!")


if __name__ == "__main__":
    main()