# 🚀 AutoGradeNet: A Novel Fully Autonomous Machine Learning Framework for Human‑Free Multi‑Dimensional Student Assessment and Academic Performance Prediction

<p align="center">
<img src="https://img.shields.io/badge/AutoGradeNet-AI%20Driven%20Assessment-blueviolet?style=for-the-badge" alt="AutoGradeNet">
<img src="https://img.shields.io/badge/Developed%20By-Dr.%20Sanjay%20Agal-darkred?style=for-the-badge" alt="Developed By">
<img src="https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python" alt="Python">
<img src="https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen?style=for-the-badge" alt="Machine Learning">
<img src="https://img.shields.io/badge/Statistics-Integrated-orange?style=for-the-badge" alt="Statistics">
<img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License">
</p>

---

## 👨‍💻 Developed By

**Dr. Sanjay Agal**  
*Head of Department, Artificial Intelligence and Data Science*  
Parul University, India  
Researcher in Machine Learning, Statistics, Educational Analytics, and Intelligent Systems

---

## 🎯 Overview

**AutoGradeNet** is an advanced **fully autonomous machine learning framework** designed for **human‑free student assessment**, **multi‑dimensional academic analytics**, and **performance prediction**. The framework eliminates manual intervention by intelligently analysing academic, behavioural, attendance, and skill‑based indicators using predictive intelligence.

✨ **Ideal for:**  
- Universities & Smart Campuses  
- EdTech Platforms & Examination Boards  
- Learning Management Systems  

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| ✅ Fully Automated Evaluation | Zero human intervention from data ingestion to report generation. |
| ✅ Multi‑Dimensional Analysis | Assesses surface, deep, structural, and behavioural skills. |
| ✅ Real‑Time Prediction | Instant grade prediction and risk flagging. |
| ✅ Explainable AI (XAI) | Feature importance and SHAP‑based insights. |
| ✅ Fairness Monitoring | Detects bias across gender, socioeconomic, and linguistic groups. |
| ✅ Statistical Rigour | Built‑in hypothesis testing, correlation, and outlier analysis. |
| ✅ Scalable Architecture | Handles thousands of students using parallel processing. |
| ✅ Publication‑Ready | Generates metrics, figures, and reports for research dissemination. |

---

## 🧠 Core Modules

| Module | Description |
|--------|-------------|
| **📊 Data Engine** | Generates synthetic academic datasets with realistic correlations (5,000 students, 4 subjects, 16 weeks). |
| **🤖 ML Engine** | Trains multi‑output regression models (Random Forest, XGBoost, Neural Networks) to predict four score dimensions. |
| **📈 Prediction Layer** | Outputs final grades, dimension‑wise scores, and at‑risk classifications. |
| **📉 Statistical Engine** | Computes descriptive statistics, ANOVA, and correlation matrices. |
| **⚖️ Fairness Engine** | Compares model performance across demographic subgroups (e.g., gender, SES). |
| **📊 Analytics Dashboard** | Generates distribution plots, heatmaps, and feature importance graphs (saved in `figures/`). |
| **🔍 Explainability Layer** | Provides global and local feature importance using permutation importance and SHAP values. |

---

## 📐 Assessment Dimensions

| Dimension | Description | Example Indicators |
|-----------|-------------|--------------------|
| **Surface Score** | Memory‑based recall and routine performance | Quiz scores, multiple‑choice accuracy |
| **Deep Score** | Conceptual understanding and analytical reasoning | Essay arguments, problem‑solving transfer |
| **Structural Score** | Logical organisation and framework building | Report structure, code modularity |
| **Behavioural Score** | Engagement, discipline, and consistency | Attendance rate, forum posts, late submissions |

These dimensions are predicted jointly using a **multi‑output regression** architecture that exploits their natural correlations (see the correlation heatmap in `figures/`).

---

## 🧪 Methodology

### 4.1 Data Generation

A synthetic dataset of **5,000 students** is generated with realistic statistical dependencies:

- **Demographics**: Age, gender, socioeconomic status, parental education, prior GPA.
- **Assessment Scores**: 4 subjects × 4 assessment types × multi‑dimensional scores (Total, Surface, Deep, Structural, Behavioural).  
  Scores are drawn from a multivariate normal distribution with a covariance structure that mimics real educational data.
- **Learning Analytics**: Login frequency, forum posts, resource views, quiz attempts.
- **Behavioural Features**: Late submissions, skipped questions, help requests, attendance rate, average response time.
- **External Factors**: Sleep hours, stress level, parental involvement, internet quality.
- **Ground Truth Labels**: Final grade computed as a weighted sum of the four dimensions plus penalty terms:

\[
\text{FinalGrade} = 0.35\cdot S_{\text{surface}} + 0.30\cdot S_{\text{deep}} + 0.20\cdot S_{\text{structural}} + 0.15\cdot S_{\text{behavioural}} - 0.5\cdot(\text{late\_submissions} + \text{skipped\_questions}) + 0.1\cdot\text{total\_logins}
\]

The result is normalised to a 0–100 scale.

### 4.2 Model Architecture

We employ a **Multi‑Output Random Forest Regressor** with 100 trees, which jointly predicts all four dimension scores. This model:

- Handles non‑linear relationships  
- Provides built‑in feature importance  
- Scales efficiently to thousands of samples  

Alternative models (e.g., XGBoost, Multi‑Layer Perceptron) can be easily substituted in `src/run_evaluation.py`.

## Number of students: 5000
Average Final Grade: 46.61
Average Surface Score: 18.40
Average Deep Score: 18.40
Average Structural Score: 18.42
Average Behavioural Score: 18.41


