#!/usr/bin/env python3
"""
BreastCAR — Revised Pipeline (Reviewer-Addressed)
===================================================
Addresses ALL reviewer comments requiring computation:

R1.3  → Y-randomisation (10 permutations) + Murcko scaffold-split validation
R1.4  → Applicability Domain (Tanimoto k=5 NN, 95th-percentile threshold)
R1.6 / R2.7 → SVM-RBF + MLP added alongside RF / XGB / LGB
R2.10 → MCC, Macro-F1, Balanced-Accuracy, per-class Precision/Recall/F1
R2.11 → Optuna Bayesian hyperparameter search (20 trials, Optuna-subsample ≤3000)
"""

import os, sys, json, logging, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, cross_val_score)
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score,
    precision_score, recall_score,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE        = Path("/home/stalin/Desktop/Breast_Cancer_ML")
DATA_DIR    = BASE / "NEW"
OUT         = BASE / "revised_results"
MODELS_REG  = OUT / "models" / "regression"
MODELS_CLF  = OUT / "models" / "classification"
RESULTS_DIR = OUT / "results"
FIGS_DIR    = OUT / "figures"

for p in [MODELS_REG, MODELS_CLF, RESULTS_DIR, FIGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_log_file = str(BASE / 'revised_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[_FlushFileHandler(_log_file, mode='w')]
)
log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACTIVE_THR   = 7.0
MODERATE_THR = 5.0
MIN_SAMPLES  = 80
FP_RADIUS    = 2
FP_NBITS     = 2048
TEST_SIZE    = 0.2
RS           = 42
CV_FOLDS     = 5
MAX_TRAIN    = 6000      # subsample cap for training
OPTUNA_CAP   = 3000      # subsample cap during Optuna search only
N_TRIALS     = 20        # Optuna trials per algorithm
N_YRANDOM    = 10        # Y-randomisation permutations

RDKIT_DESC = [
    'MolLogP','MolMR','TPSA','NumHAcceptors','NumHDonors',
    'NumRotatableBonds','RingCount','NumAromaticRings',
    'FractionCSP3','HeavyAtomCount','NumHeteroatoms','BalabanJ','BertzCT'
]
LABEL_NAMES = {0:'Inactive', 1:'Moderate', 2:'Active'}

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def clean_smiles(s):
    s = str(s).strip()
    if ' |' in s:
        s = s[:s.index(' |')].strip()
    return s

def load_target(target):
    td = DATA_DIR / target
    if not td.exists():
        return None
    csvs = sorted([f for f in td.glob('*.csv') if not f.name.startswith('._')])
    if not csvs:
        return None
    # prefer file whose name matches target (e.g. AKT1.csv over labels.csv)
    main = [f for f in csvs if target.lower() in f.stem.lower() or
            any(x in f.stem.lower() for x in ['akt','bcl','egfr','jak','kras',
                'braf','mmp','mtor','cdk','her','fgfr','parp','akt',
                'aromatase','andr','aura','brd','ca9','ccnd','ccne',
                'chk','cxcr','erbb','hdac','ldha','mapk','mdm','nfkb',
                'notc','oest','pd1','pk3','ppar','prog','rasn','s1pr',
                'stat','tgfr','tlr','tnf','tp53','vgfr','wee','pcd'])]
    csv = main[0] if main else csvs[0]

    df = pd.read_csv(csv, usecols=lambda c: not c.startswith('Unnamed'))
    # find pIC50 column
    pic50_col = None
    for c in ['pIC50', 'Activity', 'activity', 'IC50', 'pic50']:
        if c in df.columns:
            pic50_col = c
            break
    if pic50_col is None or 'SMILES' not in df.columns:
        return None

    df = df[['SMILES', pic50_col]].rename(columns={pic50_col: 'pIC50'}).copy()
    df['pIC50'] = pd.to_numeric(df['pIC50'], errors='coerce')
    df = df.dropna(subset=['pIC50'])
    df = df[(df['pIC50'] >= 2.0) & (df['pIC50'] <= 14.0)]

    # validate SMILES
    df['SMILES'] = df['SMILES'].apply(clean_smiles)
    valid = [Chem.MolFromSmiles(str(s)) is not None for s in df['SMILES']]
    df = df[valid].drop_duplicates(subset='SMILES').reset_index(drop=True)

    df['Activity'] = df['pIC50'].apply(
        lambda x: 'Active' if x >= ACTIVE_THR else ('Moderate' if x >= MODERATE_THR else 'Inactive'))
    # If any class is empty, fall back to tertile-based thresholds
    if df['Activity'].nunique() < 3:
        t33, t67 = df['pIC50'].quantile([1/3, 2/3])
        df['Activity'] = df['pIC50'].apply(
            lambda x: 'Active' if x >= t67 else ('Moderate' if x >= t33 else 'Inactive'))
    df['Label'] = df['Activity'].map({'Inactive': 0, 'Moderate': 1, 'Active': 2})
    return df

# ─── FEATURISATION ────────────────────────────────────────────────────────────
def featurize(smiles_list):
    fps, descs, mask = [], [], []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError
            fp = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, FP_RADIUS, nBits=FP_NBITS), np.float32)
            d = []
            for dn in RDKIT_DESC:
                try:
                    d.append(float(getattr(Descriptors, dn)(mol)))
                except:
                    d.append(0.0)
            fps.append(fp); descs.append(np.array(d, np.float32)); mask.append(True)
        except:
            fps.append(np.zeros(FP_NBITS, np.float32))
            descs.append(np.zeros(len(RDKIT_DESC), np.float32))
            mask.append(False)
    return np.hstack([np.vstack(fps), np.vstack(descs)]), np.array(mask)

# ─── SCAFFOLD SPLIT ───────────────────────────────────────────────────────────
def scaffold_split(smiles, labels, test_size=TEST_SIZE, random_state=RS):
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            sca = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else '__none__'
        except:
            sca = '__none__'
        scaffolds[sca].append(i)

    rng    = np.random.RandomState(random_state)
    groups = list(scaffolds.values())
    rng.shuffle(groups)

    n_test = int(len(smiles) * test_size)
    test_idx, train_idx = [], []
    for g in groups:
        if len(test_idx) < n_test:
            test_idx.extend(g)
        else:
            train_idx.extend(g)
    return np.array(train_idx), np.array(test_idx)

# ─── APPLICABILITY DOMAIN ─────────────────────────────────────────────────────
def build_ad(X_train_fp, k=5, percentile=95):
    """Return (train_fps_rdkit, threshold) for Tanimoto-based AD."""
    n = len(X_train_fp)
    fps = []
    for row in X_train_fp:
        bv = DataStructs.ExplicitBitVect(FP_NBITS)
        for j, v in enumerate(row.astype(int)):
            if v:
                bv.SetBit(j)
        fps.append(bv)

    sims = []
    for i in range(n):
        others = [fps[j] for j in range(n) if j != i]
        s = DataStructs.BulkTanimotoSimilarity(fps[i], others)
        top = sorted(s, reverse=True)[:k]
        sims.append(np.mean(top) if top else 0.0)

    threshold = float(np.percentile(sims, 100 - percentile))
    return fps, threshold

def in_ad(query_fp_row, train_fps, threshold, k=5):
    bv = DataStructs.ExplicitBitVect(FP_NBITS)
    for j, v in enumerate(query_fp_row.astype(int)):
        if v:
            bv.SetBit(j)
    s = DataStructs.BulkTanimotoSimilarity(bv, train_fps)
    top = sorted(s, reverse=True)[:k]
    return float(np.mean(top)) >= threshold if top else False

# ─── METRICS ──────────────────────────────────────────────────────────────────
def reg_metrics(y_true, y_pred):
    return dict(
        r2   = round(float(r2_score(y_true, y_pred)), 4),
        rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        mae  = round(float(mean_absolute_error(y_true, y_pred)), 4),
    )

def clf_metrics_full(y_true, y_pred, y_prob=None):
    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    m = dict(
        accuracy          = round(float(accuracy_score(y_true, y_pred)), 4),
        balanced_accuracy = round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        f1_weighted       = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        f1_macro          = round(float(f1_score(y_true, y_pred, average='macro',    zero_division=0)), 4),
        mcc               = round(float(matthews_corrcoef(y_true, y_pred)), 4),
        precision_weighted= round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        recall_weighted   = round(float(recall_score(y_true, y_pred, average='weighted',    zero_division=0)), 4),
    )
    for lab in present:
        nm  = LABEL_NAMES.get(lab, str(lab))
        mt  = (y_true == lab); mp = (y_pred == lab)
        tp  = int((mt & mp).sum()); fp = int((~mt & mp).sum()); fn = int((mt & ~mp).sum())
        pr  = tp/(tp+fp) if (tp+fp) else 0.0
        rc  = tp/(tp+fn) if (tp+fn) else 0.0
        f1c = 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
        m[f'precision_{nm}'] = round(pr,  4)
        m[f'recall_{nm}']    = round(rc,  4)
        m[f'f1_{nm}']        = round(f1c, 4)
    if y_prob is not None:
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(y_true, y_prob[:,1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            m['auc_roc'] = round(float(auc), 4)
        except:
            m['auc_roc'] = None
    return m

# ─── Y-RANDOMISATION ──────────────────────────────────────────────────────────
def _yrandom_one(X_tr, y_tr, X_te, y_te, model_fn, task, seed):
    rng = np.random.RandomState(seed)
    yp  = rng.permutation(y_tr)
    m   = clone(model_fn); m.fit(X_tr, yp)
    pred = m.predict(X_te)
    if task == 'reg':
        return r2_score(y_te, pred)
    return f1_score(y_te, pred, average='weighted', zero_division=0)

def y_randomize(X_tr, y_tr, X_te, y_te, model_fn, task='reg', n=N_YRANDOM):
    scores = joblib.Parallel(n_jobs=n, prefer='threads')(
        joblib.delayed(_yrandom_one)(X_tr, y_tr, X_te, y_te, model_fn, task, RS + 777 + i)
        for i in range(n)
    )
    return round(float(np.mean(scores)), 4), round(float(np.std(scores)), 4)

# ─── OPTUNA SEARCH ────────────────────────────────────────────────────────────
def _subsample(X, y, cap):
    if len(X) <= cap:
        return X, y
    rng  = np.random.RandomState(RS)
    idx  = rng.choice(len(X), cap, replace=False)
    return X[idx], y[idx]

def optuna_search_reg(X_tr, y_tr, algo):
    Xs, ys = _subsample(X_tr, y_tr, OPTUNA_CAP)
    kf     = KFold(n_splits=5, shuffle=True, random_state=RS)

    def objective(trial):
        m = _make_reg(algo, trial)
        return float(np.mean(
            cross_val_score(m, Xs, ys, cv=kf, scoring='r2', n_jobs=5)
        ))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params

def optuna_search_clf(X_tr, y_tr, algo):
    Xs, ys = _subsample(X_tr, y_tr, OPTUNA_CAP)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    _f1w   = make_scorer(f1_score, average='weighted', zero_division=0)

    def objective(trial):
        m = _make_clf(algo, trial)
        return float(np.mean(
            cross_val_score(m, Xs, ys, cv=skf, scoring=_f1w, n_jobs=5)
        ))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RS))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params

def _make_reg(algo, trial_or_params):
    t = trial_or_params
    is_trial = hasattr(t, 'suggest_int')
    def gi(k, lo, hi): return t.suggest_int(k, lo, hi) if is_trial else t[k]
    def gf(k, lo, hi, log=False): return t.suggest_float(k, lo, hi, log=log) if is_trial else t[k]
    def gc(k, ch): return t.suggest_categorical(k, ch) if is_trial else t[k]

    if algo == 'RF':
        return RandomForestRegressor(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 20),
            max_features    = gc('max_features', ['sqrt','log2']),
            min_samples_leaf= gi('min_samples_leaf', 1, 8),
            n_jobs=-1, random_state=RS)
    if algo == 'XGB':
        return xgb.XGBRegressor(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 10),
            learning_rate   = gf('learning_rate', 0.01, 0.3, True),
            subsample       = gf('subsample', 0.5, 1.0),
            colsample_bytree= gf('colsample_bytree', 0.5, 1.0),
            reg_alpha       = gf('reg_alpha', 1e-4, 10.0, True),
            reg_lambda      = gf('reg_lambda', 1e-4, 10.0, True),
            n_jobs=15, tree_method='hist', device='cpu',
            random_state=RS, verbosity=0)
    if algo == 'LGB':
        return lgb.LGBMRegressor(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 12),
            learning_rate   = gf('learning_rate', 0.01, 0.3, True),
            num_leaves      = gi('num_leaves', 16, 128),
            subsample       = gf('subsample', 0.5, 1.0),
            colsample_bytree= gf('colsample_bytree', 0.5, 1.0),
            reg_alpha       = gf('reg_alpha', 1e-4, 10.0, True),
            reg_lambda      = gf('reg_lambda', 1e-4, 10.0, True),
            n_jobs=-1, random_state=RS, verbose=-1)

def _make_clf(algo, trial_or_params):
    t = trial_or_params
    is_trial = hasattr(t, 'suggest_int')
    def gi(k, lo, hi): return t.suggest_int(k, lo, hi) if is_trial else t[k]
    def gf(k, lo, hi, log=False): return t.suggest_float(k, lo, hi, log=log) if is_trial else t[k]
    def gc(k, ch): return t.suggest_categorical(k, ch) if is_trial else t[k]

    if algo == 'RF':
        return RandomForestClassifier(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 20),
            max_features    = gc('max_features', ['sqrt','log2']),
            min_samples_leaf= gi('min_samples_leaf', 1, 8),
            class_weight='balanced', n_jobs=-1, random_state=RS)
    if algo == 'XGB':
        return xgb.XGBClassifier(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 10),
            learning_rate   = gf('learning_rate', 0.01, 0.3, True),
            subsample       = gf('subsample', 0.5, 1.0),
            colsample_bytree= gf('colsample_bytree', 0.5, 1.0),
            reg_alpha       = gf('reg_alpha', 1e-4, 10.0, True),
            reg_lambda      = gf('reg_lambda', 1e-4, 10.0, True),
            n_jobs=15, tree_method='hist', device='cpu',
            random_state=RS, verbosity=0)
    if algo == 'LGB':
        return lgb.LGBMClassifier(
            n_estimators    = gi('n_estimators', 100, 800),
            max_depth       = gi('max_depth', 3, 12),
            learning_rate   = gf('learning_rate', 0.01, 0.3, True),
            num_leaves      = gi('num_leaves', 16, 128),
            subsample       = gf('subsample', 0.5, 1.0),
            colsample_bytree= gf('colsample_bytree', 0.5, 1.0),
            reg_alpha       = gf('reg_alpha', 1e-4, 10.0, True),
            reg_lambda      = gf('reg_lambda', 1e-4, 10.0, True),
            class_weight='balanced', n_jobs=-1, random_state=RS, verbose=-1)

# ─── PER-TARGET PIPELINE ──────────────────────────────────────────────────────
def run_target(target):
    log.info(f"\n{'='*60}\nTARGET: {target}")

    df = load_target(target)
    if df is None or len(df) < MIN_SAMPLES:
        log.warning(f"  SKIP — n={len(df) if df is not None else 0}")
        return None

    n = len(df)
    log.info(f"  n={n}  Active={(df.Activity=='Active').sum()}  "
             f"Moderate={(df.Activity=='Moderate').sum()}  "
             f"Inactive={(df.Activity=='Inactive').sum()}")

    # Featurise
    X_all, mask = featurize(df['SMILES'].tolist())
    smiles_all  = np.array(df['SMILES'].tolist())[mask]
    y_reg_all   = df['pIC50'].values[mask]
    y_clf_all   = df['Label'].values[mask]
    X_all       = X_all[mask]

    # Subsample large datasets for training
    if len(X_all) > int(MAX_TRAIN / (1 - TEST_SIZE)):
        rng  = np.random.RandomState(RS)
        keep = rng.choice(len(X_all), int(MAX_TRAIN / (1 - TEST_SIZE)), replace=False)
        X_all = X_all[keep]; y_reg_all = y_reg_all[keep]
        y_clf_all = y_clf_all[keep]; smiles_all = smiles_all[keep]
        log.info(f"  Subsampled to {len(X_all)}")

    if len(X_all) < MIN_SAMPLES:
        log.warning(f"  SKIP after featurise — n={len(X_all)}")
        return None

    # Random split
    idx_tr, idx_te = train_test_split(
        np.arange(len(X_all)), test_size=TEST_SIZE,
        random_state=RS, stratify=y_clf_all)
    X_tr, X_te   = X_all[idx_tr], X_all[idx_te]
    yr_tr, yr_te = y_reg_all[idx_tr], y_reg_all[idx_te]
    yc_tr, yc_te = y_clf_all[idx_tr], y_clf_all[idx_te]

    # Scaffold split
    sc_tr_idx, sc_te_idx = scaffold_split(smiles_all, y_clf_all)
    Xs_tr, Xs_te         = X_all[sc_tr_idx], X_all[sc_te_idx]
    yrs_tr, yrs_te       = y_reg_all[sc_tr_idx], y_reg_all[sc_te_idx]
    ycs_tr, ycs_te       = y_clf_all[sc_tr_idx], y_clf_all[sc_te_idx]

    # Scaler for SVM / MLP
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr)
    X_te_sc   = scaler.transform(X_te)
    Xs_tr_sc  = scaler.transform(Xs_tr)
    Xs_te_sc  = scaler.transform(Xs_te)

    meta = {'target': target, 'n': len(X_all),
            'n_train': len(X_tr), 'n_test': len(X_te),
            'n_scaffold_train': len(sc_tr_idx), 'n_scaffold_test': len(sc_te_idx)}

    # ── REGRESSION ────────────────────────────────────────────────────────────
    log.info("  [REG] Optuna search (RF / XGB / LGB)...")
    reg_records  = {}
    best_hp_reg  = {}

    for algo in ['RF', 'XGB', 'LGB']:
        log.info(f"    {algo}  ({N_TRIALS} trials)")
        params = optuna_search_reg(X_tr, yr_tr, algo)
        best_hp_reg[algo] = params
        model  = _make_reg(algo, params)
        model.fit(X_tr, yr_tr)
        yp     = model.predict(X_te)
        # 5-fold CV on full training data
        kf     = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
        cv_r2  = [r2_score(yr_tr[va], clone(model).fit(X_tr[tr], yr_tr[tr]).predict(X_tr[va]))
                  for tr, va in kf.split(X_tr)]
        m = reg_metrics(yr_te, yp)
        m.update(cv_r2_mean=round(float(np.mean(cv_r2)),4),
                 cv_r2_std =round(float(np.std(cv_r2)), 4),
                 best_params=params, y_pred=yp.tolist())
        reg_records[algo] = m
        log.info(f"      R²={m['r2']:.3f}  RMSE={m['rmse']:.3f}  CV_R²={m['cv_r2_mean']:.3f}±{m['cv_r2_std']:.3f}")

    # SVM regression
    log.info("    SVM-RBF")
    svm_r = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    svm_r.fit(X_tr_sc, yr_tr)
    yp_sv = svm_r.predict(X_te_sc)
    kf    = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
    cv_sv = cross_val_score(clone(svm_r), X_tr_sc, yr_tr, cv=kf, scoring='r2', n_jobs=1)
    m_sv  = reg_metrics(yr_te, yp_sv)
    m_sv.update(cv_r2_mean=round(float(np.mean(cv_sv)),4), cv_r2_std=round(float(np.std(cv_sv)),4), y_pred=yp_sv.tolist())
    reg_records['SVM'] = m_sv
    log.info(f"      R²={m_sv['r2']:.3f}  RMSE={m_sv['rmse']:.3f}")

    # MLP regression
    log.info("    MLP")
    mlp_r = MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=500,
                         early_stopping=True, random_state=RS)
    mlp_r.fit(X_tr_sc, yr_tr)
    yp_ml = mlp_r.predict(X_te_sc)
    cv_ml = cross_val_score(clone(mlp_r), X_tr_sc, yr_tr, cv=kf, scoring='r2', n_jobs=5)
    m_ml  = reg_metrics(yr_te, yp_ml)
    m_ml.update(cv_r2_mean=round(float(np.mean(cv_ml)),4), cv_r2_std=round(float(np.std(cv_ml)),4), y_pred=yp_ml.tolist())
    reg_records['MLP'] = m_ml
    log.info(f"      R²={m_ml['r2']:.3f}  RMSE={m_ml['rmse']:.3f}")

    # Best tree model
    best_reg_algo   = max(['RF','XGB','LGB'], key=lambda a: reg_records[a]['r2'])
    best_reg_model  = _make_reg(best_reg_algo, best_hp_reg[best_reg_algo])
    best_reg_model.fit(X_tr, yr_tr)
    joblib.dump(best_reg_model, MODELS_REG / f'{target}_best_reg.pkl')

    # Y-randomisation (regression)
    log.info(f"  [Y-RAND REG]  {N_YRANDOM} permutations")
    yr_m, yr_s = y_randomize(X_tr, yr_tr, X_te, yr_te,
                              _make_reg(best_reg_algo, best_hp_reg[best_reg_algo]),
                              task='reg', n=N_YRANDOM)
    meta['yrandom_reg_mean'] = yr_m; meta['yrandom_reg_std'] = yr_s
    log.info(f"    Permuted R²={yr_m:.3f}±{yr_s:.3f}  (real={reg_records[best_reg_algo]['r2']:.3f})")

    # Scaffold-split regression
    if len(np.unique(ycs_tr)) >= 2 and len(sc_te_idx) >= 5:
        log.info("  [SCAFFOLD REG]")
        sc_reg = _make_reg(best_reg_algo, best_hp_reg[best_reg_algo])
        sc_reg.fit(Xs_tr, yrs_tr)
        sc_m = reg_metrics(yrs_te, sc_reg.predict(Xs_te))
        meta['scaffold_reg_r2']   = sc_m['r2']
        meta['scaffold_reg_rmse'] = sc_m['rmse']
        log.info(f"    Scaffold R²={sc_m['r2']:.3f}  RMSE={sc_m['rmse']:.3f}")

    meta['reg'] = reg_records

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    if len(np.unique(yc_tr)) < 2:
        log.warning("  Only 1 class — skip classification")
        meta['clf'] = None
    else:
        log.info("  [CLF] Optuna search (RF / XGB / LGB)...")
        clf_records = {}
        best_hp_clf = {}

        for algo in ['RF', 'XGB', 'LGB']:
            log.info(f"    {algo}  ({N_TRIALS} trials)")
            params = optuna_search_clf(X_tr, yc_tr, algo)
            best_hp_clf[algo] = params
            model  = _make_clf(algo, params)
            model.fit(X_tr, yc_tr)
            yp     = model.predict(X_te)
            yprob  = model.predict_proba(X_te) if hasattr(model,'predict_proba') else None
            skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
            cv_f1  = [f1_score(yc_tr[va], clone(model).fit(X_tr[tr], yc_tr[tr]).predict(X_tr[va]),
                               average='weighted', zero_division=0)
                      for tr, va in skf.split(X_tr, yc_tr)]
            m = clf_metrics_full(yc_te, yp, yprob)
            m.update(cv_f1_mean=round(float(np.mean(cv_f1)),4),
                     cv_f1_std =round(float(np.std(cv_f1)), 4),
                     best_params=params, y_pred=yp.tolist())
            clf_records[algo] = m
            log.info(f"      F1={m['f1_weighted']:.3f}  MCC={m['mcc']:.3f}  BalAcc={m['balanced_accuracy']:.3f}")

        # SVM classification
        log.info("    SVM-RBF")
        svm_c = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
                    probability=True, random_state=RS)
        svm_c.fit(X_tr_sc, yc_tr)
        yp_sv  = svm_c.predict(X_te_sc)
        yprob_sv = svm_c.predict_proba(X_te_sc)
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
        _f1w = make_scorer(f1_score, average='weighted', zero_division=0)
        cv_svc = cross_val_score(clone(svm_c), X_tr_sc, yc_tr, cv=skf, scoring=_f1w, n_jobs=1)
        m_svc = clf_metrics_full(yc_te, yp_sv, yprob_sv)
        m_svc.update(cv_f1_mean=round(float(np.mean(cv_svc)),4), cv_f1_std=round(float(np.std(cv_svc)),4), y_pred=yp_sv.tolist())
        clf_records['SVM'] = m_svc
        log.info(f"      F1={m_svc['f1_weighted']:.3f}  MCC={m_svc['mcc']:.3f}")

        # MLP classification
        log.info("    MLP")
        mlp_c = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=500,
                              early_stopping=True, random_state=RS)
        mlp_c.fit(X_tr_sc, yc_tr)
        yp_ml  = mlp_c.predict(X_te_sc)
        yprob_ml = mlp_c.predict_proba(X_te_sc)
        cv_mlc = cross_val_score(clone(mlp_c), X_tr_sc, yc_tr, cv=skf, scoring=_f1w, n_jobs=5)
        m_mlc = clf_metrics_full(yc_te, yp_ml, yprob_ml)
        m_mlc.update(cv_f1_mean=round(float(np.mean(cv_mlc)),4), cv_f1_std=round(float(np.std(cv_mlc)),4), y_pred=yp_ml.tolist())
        clf_records['MLP'] = m_mlc
        log.info(f"      F1={m_mlc['f1_weighted']:.3f}  MCC={m_mlc['mcc']:.3f}")

        # Best tree model
        best_clf_algo  = max(['RF','XGB','LGB'], key=lambda a: clf_records[a]['f1_weighted'])
        best_clf_model = _make_clf(best_clf_algo, best_hp_clf[best_clf_algo])
        best_clf_model.fit(X_tr, yc_tr)
        joblib.dump(best_clf_model, MODELS_CLF / f'{target}_best_clf.pkl')

        # Y-randomisation (classification)
        log.info(f"  [Y-RAND CLF]  {N_YRANDOM} permutations")
        yr_mc, yr_sc = y_randomize(X_tr, yc_tr, X_te, yc_te,
                                   _make_clf(best_clf_algo, best_hp_clf[best_clf_algo]),
                                   task='clf', n=N_YRANDOM)
        meta['yrandom_clf_mean'] = yr_mc; meta['yrandom_clf_std'] = yr_sc
        log.info(f"    Permuted F1={yr_mc:.3f}±{yr_sc:.3f}  (real={clf_records[best_clf_algo]['f1_weighted']:.3f})")

        # Scaffold-split classification
        if len(np.unique(ycs_tr)) >= 2 and len(sc_te_idx) >= 5:
            log.info("  [SCAFFOLD CLF]")
            sc_clf = _make_clf(best_clf_algo, best_hp_clf[best_clf_algo])
            sc_clf.fit(Xs_tr, ycs_tr)
            yp_sc_c = sc_clf.predict(Xs_te)
            sc_cm   = clf_metrics_full(ycs_te, yp_sc_c)
            meta['scaffold_clf_f1']  = sc_cm['f1_weighted']
            meta['scaffold_clf_mcc'] = sc_cm['mcc']
            log.info(f"    Scaffold F1={sc_cm['f1_weighted']:.3f}  MCC={sc_cm['mcc']:.3f}")

        meta['clf'] = clf_records

    # ── APPLICABILITY DOMAIN ───────────────────────────────────────────────────
    log.info("  [AD] Computing Tanimoto-based applicability domain...")
    train_fps, ad_thresh = build_ad(X_tr[:, :FP_NBITS])
    test_in_ad = [in_ad(X_te[i, :FP_NBITS], train_fps, ad_thresh) for i in range(len(X_te))]
    pct_in_ad  = round(100 * float(sum(test_in_ad)) / len(test_in_ad), 1)
    meta['ad_threshold']      = round(ad_thresh, 4)
    meta['ad_pct_test_in_ad'] = pct_in_ad
    log.info(f"    AD threshold={ad_thresh:.4f}  {pct_in_ad}% test compounds in AD")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    save = {'meta': {k: v for k, v in meta.items() if k not in ('reg','clf')}}
    for task_key in ('reg','clf'):
        recs = meta.get(task_key)
        if recs is None:
            save[task_key] = None
            continue
        save[task_key] = {}
        for algo, m in recs.items():
            save[task_key][algo] = {k: v for k, v in m.items() if k != 'y_pred'}

    (RESULTS_DIR / f'{target}_results.json').write_text(json.dumps(save, indent=2))
    log.info(f"  Saved → {target}_results.json")
    return meta

# ─── SUMMARY TABLES ───────────────────────────────────────────────────────────
def build_summaries(all_res):
    reg_rows, clf_rows, hp_rows = [], [], []
    for r in all_res:
        if r is None:
            continue
        target = r['target']

        # Regression summary
        recs = r.get('reg')
        if recs:
            best = max(['RF','XGB','LGB'], key=lambda a: recs.get(a,{}).get('r2',-999))
            br   = recs[best]
            row  = dict(target=target, best_model=best,
                        r2=br['r2'], rmse=br['rmse'], mae=br['mae'],
                        cv_r2_mean=br.get('cv_r2_mean'), cv_r2_std=br.get('cv_r2_std'),
                        yrandom_r2_mean=r.get('yrandom_reg_mean'),
                        yrandom_r2_std =r.get('yrandom_reg_std'),
                        scaffold_r2    =r.get('scaffold_reg_r2'),
                        scaffold_rmse  =r.get('scaffold_reg_rmse'),
                        ad_threshold   =r.get('ad_threshold'),
                        ad_pct_in_ad   =r.get('ad_pct_test_in_ad'))
            for a in ['RF','XGB','LGB','SVM','MLP']:
                row[f'{a}_r2']   = recs.get(a,{}).get('r2')
                row[f'{a}_rmse'] = recs.get(a,{}).get('rmse')
            reg_rows.append(row)

            # Hyperparameter rows
            for a in ['RF','XGB','LGB']:
                if a in recs and 'best_params' in recs[a]:
                    hp_rows.append({'target': target, 'task': 'REG', 'algorithm': a,
                                    **recs[a]['best_params']})

        # Classification summary
        recs = r.get('clf')
        if recs:
            best = max(['RF','XGB','LGB'], key=lambda a: recs.get(a,{}).get('f1_weighted',-999))
            bc   = recs[best]
            row  = dict(target=target, best_model=best,
                        f1_weighted=bc['f1_weighted'], f1_macro=bc.get('f1_macro'),
                        mcc=bc.get('mcc'), balanced_accuracy=bc.get('balanced_accuracy'),
                        accuracy=bc.get('accuracy'), auc_roc=bc.get('auc_roc'),
                        cv_f1_mean=bc.get('cv_f1_mean'), cv_f1_std=bc.get('cv_f1_std'),
                        yrandom_f1_mean=r.get('yrandom_clf_mean'),
                        yrandom_f1_std =r.get('yrandom_clf_std'),
                        scaffold_f1    =r.get('scaffold_clf_f1'),
                        scaffold_mcc   =r.get('scaffold_clf_mcc'))
            for a in ['RF','XGB','LGB','SVM','MLP']:
                row[f'{a}_f1']  = recs.get(a,{}).get('f1_weighted')
                row[f'{a}_mcc'] = recs.get(a,{}).get('mcc')
            # Per-class for best model
            for cls in ['Active','Moderate','Inactive']:
                for met in ['precision','recall','f1']:
                    row[f'{best}_{met}_{cls}'] = bc.get(f'{met}_{cls}')
            clf_rows.append(row)

            for a in ['RF','XGB','LGB']:
                if a in recs and 'best_params' in recs[a]:
                    hp_rows.append({'target': target, 'task': 'CLF', 'algorithm': a,
                                    **recs[a]['best_params']})

    return pd.DataFrame(reg_rows), pd.DataFrame(clf_rows), pd.DataFrame(hp_rows)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("BreastCAR REVISED PIPELINE — All Reviewer Comments Addressed")
    log.info(f"Data: {DATA_DIR}   Output: {OUT}")
    log.info("=" * 65)

    targets = sorted([
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name not in ('ALL_TXT','Others','Non_BC_Targets','.DS_Store','BRCA1')
        and not d.name.startswith('.')
    ])
    log.info(f"Found {len(targets)} target folders")

    done = {f.stem.replace('_results','') for f in RESULTS_DIR.glob('*_results.json')}
    log.info(f"Already done: {len(done)}")

    all_res = []
    # Load existing
    for d in done:
        js = json.loads((RESULTS_DIR / f'{d}_results.json').read_text())
        r  = {'target': d, **js.get('meta',{}), 'reg': js.get('reg'), 'clf': js.get('clf')}
        all_res.append(r)

    for target in targets:
        if target in done:
            log.info(f"[SKIP] {target}")
            continue
        try:
            res = run_target(target)
            if res:
                all_res.append(res)
        except Exception as e:
            log.error(f"[FAIL] {target}: {e}", exc_info=True)

    # Summaries
    log.info("\nBuilding summary tables...")
    reg_df, clf_df, hp_df = build_summaries(all_res)

    reg_df.to_csv(RESULTS_DIR / 'regression_summary.csv', index=False)
    clf_df.to_csv(RESULTS_DIR / 'classification_summary.csv', index=False)
    hp_df.to_csv (RESULTS_DIR / 'hyperparameter_table.csv',   index=False)

    log.info("\n" + "="*65)
    log.info("DONE")
    if not reg_df.empty:
        log.info(f"  Reg  — targets={len(reg_df)}  mean R²={reg_df['r2'].mean():.3f}  mean RMSE={reg_df['rmse'].mean():.3f}")
        log.info(f"         Y-rand R² (mean)={reg_df['yrandom_r2_mean'].mean():.3f}  scaffold R²={reg_df['scaffold_r2'].mean():.3f}")
    if not clf_df.empty:
        log.info(f"  Clf  — targets={len(clf_df)}  mean F1={clf_df['f1_weighted'].mean():.3f}  mean MCC={clf_df['mcc'].mean():.3f}")
        log.info(f"         Y-rand F1 (mean)={clf_df['yrandom_f1_mean'].mean():.3f}  scaffold F1={clf_df['scaffold_f1'].mean():.3f}")
    log.info("="*65)

if __name__ == '__main__':
    main()
