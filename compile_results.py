"""
Compile all 53 per-target JSON results into summary CSV tables.
Outputs to revised_results/
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("revised_results/results")
OUT_DIR     = Path("revised_results")

MODELS = ['RF', 'XGB', 'LGB', 'SVM', 'MLP']

reg_rows  = []
clf_rows  = []
best_rows = []

for jf in sorted(RESULTS_DIR.glob("*_results.json")):
    d      = json.loads(jf.read_text())
    meta   = d['meta']
    target = meta['target']

    # ── REGRESSION ───────────────────────────────────────────────────────────
    for model in MODELS:
        if model not in d.get('reg', {}):
            continue
        r = d['reg'][model]
        reg_rows.append({
            'Target'        : target,
            'N'             : meta['n'],
            'Model'         : model,
            'R2'            : r.get('r2'),
            'RMSE'          : r.get('rmse'),
            'MAE'           : r.get('mae'),
            'CV_R2_mean'    : r.get('cv_r2_mean'),
            'CV_R2_std'     : r.get('cv_r2_std'),
            'Yrandom_R2'    : meta.get('yrandom_reg_mean'),
            'Yrandom_R2_std': meta.get('yrandom_reg_std'),
            'Scaffold_R2'   : meta.get('scaffold_reg_r2'),
            'Scaffold_RMSE' : meta.get('scaffold_reg_rmse'),
            'AD_threshold'  : meta.get('ad_threshold'),
            'AD_pct_in'     : meta.get('ad_pct_test_in_ad'),
        })

    # ── CLASSIFICATION ───────────────────────────────────────────────────────
    clf = d.get('clf')
    if clf is None:
        continue
    for model in MODELS:
        if model not in clf:
            continue
        c = clf[model]
        clf_rows.append({
            'Target'              : target,
            'N'                   : meta['n'],
            'Model'               : model,
            'Accuracy'            : c.get('accuracy'),
            'Balanced_Accuracy'   : c.get('balanced_accuracy'),
            'F1_weighted'         : c.get('f1_weighted'),
            'F1_macro'            : c.get('f1_macro'),
            'MCC'                 : c.get('mcc'),
            'AUC_ROC'             : c.get('auc_roc'),
            'CV_F1_mean'          : c.get('cv_f1_mean'),
            'CV_F1_std'           : c.get('cv_f1_std'),
            'Precision_Inactive'  : c.get('precision_Inactive'),
            'Recall_Inactive'     : c.get('recall_Inactive'),
            'F1_Inactive'         : c.get('f1_Inactive'),
            'Precision_Moderate'  : c.get('precision_Moderate'),
            'Recall_Moderate'     : c.get('recall_Moderate'),
            'F1_Moderate'         : c.get('f1_Moderate'),
            'Precision_Active'    : c.get('precision_Active'),
            'Recall_Active'       : c.get('recall_Active'),
            'F1_Active'           : c.get('f1_Active'),
            'Yrandom_F1'          : meta.get('yrandom_clf_mean'),
            'Yrandom_F1_std'      : meta.get('yrandom_clf_std'),
            'Scaffold_F1'         : meta.get('scaffold_clf_f1'),
            'Scaffold_MCC'        : meta.get('scaffold_clf_mcc'),
            'AD_threshold'        : meta.get('ad_threshold'),
            'AD_pct_in'           : meta.get('ad_pct_test_in_ad'),
        })

    # ── BEST MODEL PER TARGET ────────────────────────────────────────────────
    best_reg_model = max(
        [m for m in MODELS if m in d.get('reg', {})],
        key=lambda m: d['reg'][m].get('r2', -999)
    )
    best_clf_model = max(
        [m for m in MODELS if m in clf],
        key=lambda m: clf[m].get('f1_weighted', -999)
    )
    br = d['reg'][best_reg_model]
    bc = clf[best_clf_model]
    best_rows.append({
        'Target'            : target,
        'N'                 : meta['n'],
        # Regression
        'Best_Reg_Model'    : best_reg_model,
        'R2'                : br.get('r2'),
        'RMSE'              : br.get('rmse'),
        'MAE'               : br.get('mae'),
        'CV_R2'             : br.get('cv_r2_mean'),
        'Yrandom_R2'        : meta.get('yrandom_reg_mean'),
        'Scaffold_R2'       : meta.get('scaffold_reg_r2'),
        # Classification
        'Best_Clf_Model'    : best_clf_model,
        'F1_weighted'       : bc.get('f1_weighted'),
        'F1_macro'          : bc.get('f1_macro'),
        'MCC'               : bc.get('mcc'),
        'Balanced_Accuracy' : bc.get('balanced_accuracy'),
        'AUC_ROC'           : bc.get('auc_roc'),
        'CV_F1'             : bc.get('cv_f1_mean'),
        'Yrandom_F1'        : meta.get('yrandom_clf_mean'),
        'Scaffold_F1'       : meta.get('scaffold_clf_f1'),
        'AD_pct_in'         : meta.get('ad_pct_test_in_ad'),
    })

# ── SAVE CSVs ────────────────────────────────────────────────────────────────
df_reg  = pd.DataFrame(reg_rows)
df_clf  = pd.DataFrame(clf_rows)
df_best = pd.DataFrame(best_rows)

df_reg.to_csv(OUT_DIR / "regression_summary.csv",      index=False)
df_clf.to_csv(OUT_DIR / "classification_summary.csv",  index=False)
df_best.to_csv(OUT_DIR / "best_model_summary.csv",     index=False)

# ── PRINT OVERALL STATS ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RESULTS COMPILED — {len(df_best)} targets")
print(f"{'='*60}")

print("\n── REGRESSION (best model per target) ──")
print(f"  R²       mean={df_best['R2'].mean():.3f}  median={df_best['R2'].median():.3f}  "
      f"min={df_best['R2'].min():.3f}  max={df_best['R2'].max():.3f}")
print(f"  RMSE     mean={df_best['RMSE'].mean():.3f}  median={df_best['RMSE'].median():.3f}")
print(f"  Y-rand R² mean={df_best['Yrandom_R2'].mean():.3f}  (gap={df_best['R2'].mean()-df_best['Yrandom_R2'].mean():.3f})")
print(f"  Scaffold R² mean={df_best['Scaffold_R2'].mean():.3f}")

print("\n── CLASSIFICATION (best model per target) ──")
print(f"  F1_weighted  mean={df_best['F1_weighted'].mean():.3f}  median={df_best['F1_weighted'].median():.3f}  "
      f"min={df_best['F1_weighted'].min():.3f}  max={df_best['F1_weighted'].max():.3f}")
print(f"  MCC          mean={df_best['MCC'].mean():.3f}  median={df_best['MCC'].median():.3f}")
print(f"  Bal_Acc      mean={df_best['Balanced_Accuracy'].mean():.3f}")
print(f"  AUC-ROC      mean={df_best['AUC_ROC'].mean():.3f}")
print(f"  Y-rand F1    mean={df_best['Yrandom_F1'].mean():.3f}  (gap={df_best['F1_weighted'].mean()-df_best['Yrandom_F1'].mean():.3f})")
print(f"  Scaffold F1  mean={df_best['Scaffold_F1'].mean():.3f}")
print(f"  AD coverage  mean={df_best['AD_pct_in'].mean():.1f}%")

print("\n── BEST MODEL DISTRIBUTION ──")
print("  Regression:", df_best['Best_Reg_Model'].value_counts().to_dict())
print("  Classification:", df_best['Best_Clf_Model'].value_counts().to_dict())

print("\n── TOP 5 TARGETS BY R² ──")
print(df_best[['Target','Best_Reg_Model','R2','RMSE']].sort_values('R2', ascending=False).head(5).to_string(index=False))

print("\n── TOP 5 TARGETS BY F1 ──")
print(df_best[['Target','Best_Clf_Model','F1_weighted','MCC','AUC_ROC']].sort_values('F1_weighted', ascending=False).head(5).to_string(index=False))

print(f"\nSaved:\n  {OUT_DIR}/regression_summary.csv\n  {OUT_DIR}/classification_summary.csv\n  {OUT_DIR}/best_model_summary.csv")
