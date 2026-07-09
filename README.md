# BreastCAR — Breast Cancer Drug Activity Prediction

**BreastCAR** is a breast cancer drug activity prediction framework that trains parallel single-target ensemble machine learning models across **53 targets** using ChEMBL v33 bioactivity data, and serves predictions through a public web application.

---

## Overview

| Item | Detail |
|------|--------|
| Targets | 53 breast cancer-relevant proteins |
| Compounds | 234,168 compound–activity pairs (ChEMBL v33) |
| Algorithms | RF, XGB, LGB, SVM-RBF, MLP |
| Regression | Mean R² = 0.723, RMSE = 0.661 |
| Classification | Mean F1 = 0.826, MCC = 0.679, AUC-ROC = 0.919 |
| Validation | Y-randomisation (10 perms), Murcko scaffold split, 5-fold CV, AD (Tanimoto k-NN) |

**Web app**: https://breastcar-qsar.onrender.com

---

## Repository Structure

```
BreastCAR-QSAR/
├── pipeline.py        # Main QSAR pipeline — trains all 53 targets
├── requirements.txt
├── render.yaml        # Render.com deployment config
└── webapp/
    ├── app.py         # FastAPI web application
    ├── templates/
    │   └── index.html
    └── static/
```

---

## Pipeline

### Run the QSAR pipeline

```bash
pip install -r requirements.txt
python pipeline.py
```

**Outputs** (written to `revised_results/`):
```
revised_results/
├── results/<TARGET>_results.json   # per-target results (53 files)
├── regression_summary.csv
├── classification_summary.csv
├── best_model_summary.csv
├── Supplementary_Table_S1.xlsx
└── Supplementary_Table_S2_Hyperparameters.xlsx
```

Checkpoint resume is supported — already-completed targets are skipped on restart.

---

## Web Application

The FastAPI web app allows real-time QSAR prediction for any SMILES input across all 53 targets.

### Live deployment

**https://breastcar-qsar.onrender.com**

Models (~329 MB compressed) are downloaded automatically from GitHub Releases on first startup.

### Run locally

```bash
cd webapp
uvicorn app:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

Set `MODEL_DIR` environment variable to a writable path for model storage (default: `../models/`).

### Features

- Predict pIC50 (regression) and activity class (Active / Moderate / Inactive) for all 53 targets
- Molecular structure visualisation (RDKit SVG)
- Applicability Domain (AD) status per target
- Physicochemical property panel (MW, LogP, TPSA, Lipinski Ro5, …)
- Activity distribution chart + top-15 bar chart
- Sortable / filterable predictions table
- CSV and JSON export
- Batch prediction mode
- Target library explorer

---

## Models

| Algorithm | Type | Tuning |
|-----------|------|--------|
| Random Forest (RF) | Ensemble | Optuna 20 trials, 5-fold CV |
| XGBoost (XGB) | Gradient Boosting | Optuna 20 trials, 5-fold CV |
| LightGBM (LGB) | Gradient Boosting | Optuna 20 trials, 5-fold CV |
| SVM-RBF | Kernel SVM | Fixed (C=10, γ=scale, StandardScaler) |
| MLP | Neural Network | Fixed (256-128-64, Adam, early stopping, StandardScaler) |

**Features**: ECFP4 fingerprints (2048 bits, radius=2) + 13 RDKit physicochemical descriptors = 2061-dimensional feature vector.

**Best model selection**: the algorithm with the highest cross-validated R² (regression) or F1-weighted (classification) is saved per target.

---

## Validation

| Method | Result |
|--------|--------|
| Y-randomisation (10 perms) | Permuted R² = −0.167 vs real 0.723 (gap = 0.890) |
| Y-randomisation CLF | Permuted F1 = 0.467 vs real 0.826 (gap = 0.359) |
| Murcko scaffold split | Scaffold R² = 0.587, Scaffold F1 = 0.766 |
| Applicability Domain | Tanimoto k-NN (k=5, 95th-percentile); 94.8% test-set coverage |

---

## Activity Thresholds

| Class | pIC50 | IC50 |
|-------|-------|------|
| Active | ≥ 7.0 | ≤ 100 nM |
| Moderate | 5.0 – 7.0 | 100 nM – 10 µM |
| Inactive | < 5.0 | > 10 µM |

---

## Citation

> Stalin S. et al. *BreastCAR: A Comprehensive Breast Cancer Drug Activity Prediction Framework Using Parallel Single-Target Ensemble Machine Learning Across 53 Targets.* (Under revision, 2026)

---

## License

MIT License
