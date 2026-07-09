# BreastCAR — Multi-Target QSAR Framework

**BreastCAR** is a comprehensive multi-target QSAR (Quantitative Structure–Activity Relationship) framework for breast cancer drug activity prediction across **53 targets** using ChEMBL v33 bioactivity data.

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

---

## Repository Structure

```
BreastCAR-QSAR/
├── revised_pipeline.py             # Main QSAR pipeline (all 53 targets)
├── compile_results.py              # Compile per-target JSONs → summary CSVs
├── generate_figures_tables.py      # Generate supplementary figures + Table S1
├── generate_manuscript_figures.py  # Generate manuscript figures 1–10
├── revise_manuscript.py            # Apply reviewer revisions to manuscript docx
├── requirements.txt
├── render.yaml                     # Render.com deployment config
└── webapp/
    ├── app.py                      # FastAPI web application
    ├── templates/index.html
    └── static/
```

---

## Pipeline

### 1. Run the QSAR pipeline
```bash
python revised_pipeline.py
# Outputs: revised_results/results/<TARGET>_results.json for all 53 targets
# Supports checkpoint resume — skips already-completed targets
```

### 2. Compile results
```bash
python compile_results.py
# Outputs: revised_results/regression_summary.csv
#          revised_results/classification_summary.csv
#          revised_results/best_model_summary.csv
```

### 3. Generate figures
```bash
python generate_figures_tables.py      # Supplementary figures + Table S1
python generate_manuscript_figures.py  # Manuscript figures 1–10
```

---

## Web Application

The FastAPI web app allows real-time QSAR prediction for any SMILES input across all 53 targets.

### Run locally
```bash
cd webapp
uvicorn app:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

### Features
- Input any SMILES string
- Predict pIC50 (regression) and activity class (Active / Moderate / Inactive) for all 53 targets
- Molecular structure visualisation
- Target activity profile heatmap

---

## Models

| Algorithm | Type | Tuning |
|-----------|------|--------|
| Random Forest (RF) | Ensemble | Optuna 20 trials |
| XGBoost (XGB) | Gradient Boosting | Optuna 20 trials |
| LightGBM (LGB) | Gradient Boosting | Optuna 20 trials |
| SVM-RBF | Kernel SVM | Fixed (C=10, γ=scale) |
| MLP | Neural Network | Fixed (256-128-64, early stopping) |

Features: ECFP4 fingerprints (2048 bits, r=2) + 12 RDKit physicochemical descriptors

---

## Validation

- **Y-randomisation**: permuted R² = −0.167 vs real 0.723 (gap = 0.890)
- **Scaffold split**: mean scaffold R² = 0.587, scaffold F1 = 0.766
- **Applicability Domain**: Tanimoto k-NN (k=5, 95th-percentile threshold); 94.8% test coverage

---

## Citation

> Stalin S. et al. *BreastCAR: A Comprehensive Multi-Target QSAR Framework for Breast Cancer Drug Activity Prediction Using Parallel Single-Target Ensemble Machine Learning.* (Under revision, 2026)

---

## License

MIT License
