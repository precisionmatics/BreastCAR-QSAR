"""
Generate all figures and Supplementary Table S1 for the revised manuscript.
Addresses: R1.3 (Y-rand/scaffold), R1.4 (AD), R1.6/R2.7 (SVM+MLP), R2.10 (MCC/BalAcc/Macro-F1)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("revised_results/results")
FIG_DIR     = Path("revised_results/figures")
OUT_DIR     = Path("revised_results")
FIG_DIR.mkdir(exist_ok=True)

MODELS      = ['RF', 'XGB', 'LGB', 'SVM', 'MLP']
MODEL_COLORS= {'RF':'#4C72B0','XGB':'#DD8452','LGB':'#55A868','SVM':'#C44E52','MLP':'#8172B2'}

df_reg  = pd.read_csv(OUT_DIR / "regression_summary.csv")
df_clf  = pd.read_csv(OUT_DIR / "classification_summary.csv")
df_best = pd.read_csv(OUT_DIR / "best_model_summary.csv")

targets = df_best['Target'].tolist()

# ── FIG 1: R² heatmap (all 5 models × 53 targets) ───────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
pivot_r2 = df_reg.pivot(index='Model', columns='Target', values='R2').reindex(
    index=MODELS, columns=targets)
sns.heatmap(pivot_r2, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.3, annot=False, cbar_kws={'label': 'R²'})
ax.set_title('Regression Performance (R²) — All Models × 53 Targets', fontsize=13, fontweight='bold')
ax.set_xlabel('Target', fontsize=10); ax.set_ylabel('')
ax.tick_params(axis='x', rotation=90, labelsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_r2_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 1 saved: R² heatmap")

# ── FIG 2: F1 heatmap (all 5 models × 53 targets) ───────────────────────────
fig, ax = plt.subplots(figsize=(22, 10))
pivot_f1 = df_clf.pivot(index='Model', columns='Target', values='F1_weighted').reindex(
    index=MODELS, columns=targets)
sns.heatmap(pivot_f1, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
            linewidths=0.3, annot=False, cbar_kws={'label': 'F1 (weighted)'})
ax.set_title('Classification Performance (F1 weighted) — All Models × 53 Targets', fontsize=13, fontweight='bold')
ax.set_xlabel('Target', fontsize=10); ax.set_ylabel('')
ax.tick_params(axis='x', rotation=90, labelsize=7)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_f1_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 2 saved: F1 heatmap")

# ── FIG 3: Y-randomization validation ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression
ax = axes[0]
x  = np.arange(len(targets))
ax.bar(x, df_best['R2'],         color='#4C72B0', alpha=0.85, label='Real R²')
ax.bar(x, df_best['Yrandom_R2'], color='#C44E52', alpha=0.7,  label='Y-rand R²')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.set_xticks(x); ax.set_xticklabels(targets, rotation=90, fontsize=6)
ax.set_ylabel('R²'); ax.set_title('Y-Randomization — Regression', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(-0.5, 1.0)

# Classification
ax = axes[1]
ax.bar(x, df_best['F1_weighted'], color='#4C72B0', alpha=0.85, label='Real F1')
ax.bar(x, df_best['Yrandom_F1'], color='#C44E52', alpha=0.7,  label='Y-rand F1')
ax.set_xticks(x); ax.set_xticklabels(targets, rotation=90, fontsize=6)
ax.set_ylabel('F1 (weighted)'); ax.set_title('Y-Randomization — Classification', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)

plt.suptitle('Y-Randomization Validation Across 53 Targets', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_yrandomization.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 3 saved: Y-randomization")

# ── FIG 4: Scaffold split vs random split ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(df_best['R2'], df_best['Scaffold_R2'], alpha=0.7, color='#4C72B0', edgecolors='k', lw=0.5)
lim = [min(df_best[['R2','Scaffold_R2']].min()), max(df_best[['R2','Scaffold_R2']].max())]
ax.plot(lim, lim, 'r--', lw=1, label='y=x')
for _, row in df_best.iterrows():
    ax.annotate(row['Target'], (row['R2'], row['Scaffold_R2']), fontsize=5, alpha=0.6)
ax.set_xlabel('Random Split R²'); ax.set_ylabel('Scaffold Split R²')
ax.set_title('Scaffold vs Random — Regression', fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(df_best['F1_weighted'], df_best['Scaffold_F1'], alpha=0.7, color='#55A868', edgecolors='k', lw=0.5)
lim = [min(df_best[['F1_weighted','Scaffold_F1']].min()), max(df_best[['F1_weighted','Scaffold_F1']].max())]
ax.plot(lim, lim, 'r--', lw=1, label='y=x')
for _, row in df_best.iterrows():
    ax.annotate(row['Target'], (row['F1_weighted'], row['Scaffold_F1']), fontsize=5, alpha=0.6)
ax.set_xlabel('Random Split F1'); ax.set_ylabel('Scaffold Split F1')
ax.set_title('Scaffold vs Random — Classification', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('Murcko Scaffold Split vs Random Split', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_scaffold_split.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 4 saved: scaffold split")

# ── FIG 5: MCC / Balanced Accuracy / F1-macro comparison ────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
x   = np.arange(len(targets))
w   = 0.25
b1  = ax.bar(x - w,   df_best['F1_weighted'],       w, label='F1 weighted', color='#4C72B0', alpha=0.85)
b2  = ax.bar(x,       df_best['MCC'],               w, label='MCC',         color='#DD8452', alpha=0.85)
b3  = ax.bar(x + w,   df_best['Balanced_Accuracy'], w, label='Bal. Acc.',   color='#55A868', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(targets, rotation=90, fontsize=7)
ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
ax.set_title('Classification Metrics per Target (Best Model)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_clf_metrics.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 5 saved: classification metrics")

# ── FIG 6: AD coverage ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 4))
colors = ['#C44E52' if v < 80 else '#4C72B0' for v in df_best['AD_pct_in']]
ax.bar(targets, df_best['AD_pct_in'], color=colors, alpha=0.85)
ax.axhline(80, color='red',  ls='--', lw=1, label='80% threshold')
ax.axhline(df_best['AD_pct_in'].mean(), color='navy', ls='--', lw=1,
           label=f"Mean={df_best['AD_pct_in'].mean():.1f}%")
ax.set_xticklabels(targets, rotation=90, fontsize=7)
ax.set_ylabel('% Test Compounds in AD'); ax.set_ylim(0, 105)
ax.set_title('Applicability Domain Coverage per Target (Tanimoto k=5, 95th percentile)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_ad_coverage.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 6 saved: AD coverage")

# ── FIG 7: Model selection frequency ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in zip(axes, ['Best_Reg_Model','Best_Clf_Model'],
                           ['Regression Best Model','Classification Best Model']):
    counts = df_best[col].value_counts().reindex(MODELS, fill_value=0)
    bars   = ax.bar(counts.index, counts.values,
                    color=[MODEL_COLORS[m] for m in counts.index], alpha=0.85)
    ax.set_ylabel('# Targets'); ax.set_title(title, fontweight='bold')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(v),
                ha='center', fontsize=11, fontweight='bold')
plt.suptitle('Best Model Frequency Across 53 Targets', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_model_frequency.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 7 saved: model frequency")

# ── FIG 8: Per-class F1 (Active / Moderate / Inactive) ───────────────────────
best_clf_rows = []
for jf in sorted(RESULTS_DIR.glob("*_results.json")):
    d    = json.loads(jf.read_text())
    clf  = d.get('clf')
    if clf is None: continue
    target = d['meta']['target']
    best_m = max([m for m in MODELS if m in clf], key=lambda m: clf[m].get('f1_weighted', -1))
    c      = clf[best_m]
    best_clf_rows.append({
        'Target'      : target,
        'F1_Active'   : c.get('f1_Active'),
        'F1_Moderate' : c.get('f1_Moderate'),
        'F1_Inactive' : c.get('f1_Inactive'),
    })
df_perclass = pd.DataFrame(best_clf_rows)

fig, ax = plt.subplots(figsize=(18, 5))
x  = np.arange(len(df_perclass))
w  = 0.28
ax.bar(x - w,  df_perclass['F1_Active'],   w, label='Active',   color='#4C72B0', alpha=0.85)
ax.bar(x,      df_perclass['F1_Moderate'], w, label='Moderate', color='#DD8452', alpha=0.85)
ax.bar(x + w,  df_perclass['F1_Inactive'], w, label='Inactive', color='#55A868', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(df_perclass['Target'], rotation=90, fontsize=7)
ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1.05)
ax.set_title('Per-Class F1 Score per Target (Best Model)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig8_perclass_f1.png', dpi=200, bbox_inches='tight')
plt.close()
print("Fig 8 saved: per-class F1")

# ── SUPPLEMENTARY TABLE S1 ───────────────────────────────────────────────────
rows = []
for jf in sorted(RESULTS_DIR.glob("*_results.json")):
    d    = json.loads(jf.read_text())
    meta = d['meta']
    clf  = d.get('clf')
    target = meta['target']

    best_reg = max([m for m in MODELS if m in d.get('reg', {})],
                   key=lambda m: d['reg'][m].get('r2', -999))
    br = d['reg'][best_reg]

    if clf:
        best_cl = max([m for m in MODELS if m in clf],
                      key=lambda m: clf[m].get('f1_weighted', -999))
        bc = clf[best_cl]
    else:
        best_cl = 'N/A'; bc = {}

    rows.append({
        'Target'                : target,
        'N_compounds'           : meta['n'],
        'N_train'               : meta['n_train'],
        'N_test'                : meta['n_test'],
        # Regression
        'Best_Reg_Model'        : best_reg,
        'R2'                    : round(br.get('r2', np.nan), 4),
        'RMSE'                  : round(br.get('rmse', np.nan), 4),
        'MAE'                   : round(br.get('mae', np.nan), 4),
        'CV_R2'                 : round(br.get('cv_r2_mean', np.nan), 4),
        'Yrandom_R2'            : meta.get('yrandom_reg_mean'),
        'Scaffold_R2'           : meta.get('scaffold_reg_r2'),
        # Classification
        'Best_Clf_Model'        : best_cl,
        'F1_weighted'           : round(bc.get('f1_weighted', np.nan), 4),
        'F1_macro'              : round(bc.get('f1_macro', np.nan), 4),
        'MCC'                   : round(bc.get('mcc', np.nan), 4),
        'Balanced_Accuracy'     : round(bc.get('balanced_accuracy', np.nan), 4),
        'AUC_ROC'               : round(bc.get('auc_roc', np.nan), 4),
        'CV_F1'                 : round(bc.get('cv_f1_mean', np.nan), 4),
        'Yrandom_F1'            : meta.get('yrandom_clf_mean'),
        'Scaffold_F1'           : meta.get('scaffold_clf_f1'),
        'Scaffold_MCC'          : meta.get('scaffold_clf_mcc'),
        'F1_Active'             : round(bc.get('f1_Active', np.nan), 4),
        'F1_Moderate'           : round(bc.get('f1_Moderate', np.nan), 4),
        'F1_Inactive'           : round(bc.get('f1_Inactive', np.nan), 4),
        'AD_threshold'          : meta.get('ad_threshold'),
        'AD_pct_in_AD'          : meta.get('ad_pct_test_in_ad'),
    })

df_s1 = pd.DataFrame(rows)
df_s1.to_csv(OUT_DIR / "Supplementary_Table_S1.csv", index=False)
df_s1.to_excel(OUT_DIR / "Supplementary_Table_S1.xlsx", index=False)
print(f"\nSupplementary Table S1 saved ({len(df_s1)} targets)")

# ── SUMMARY PRINT ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("ALL FIGURES & TABLES GENERATED")
print(f"{'='*55}")
print(f"Figures (8):  revised_results/figures/")
print(f"Tables (4):   revised_results/")
print(f"  regression_summary.csv      ({len(df_reg)} rows)")
print(f"  classification_summary.csv  ({len(df_clf)} rows)")
print(f"  best_model_summary.csv      ({len(df_best)} rows)")
print(f"  Supplementary_Table_S1.csv/xlsx ({len(df_s1)} targets)")
