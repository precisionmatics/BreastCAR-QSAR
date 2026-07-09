#!/usr/bin/env python3
"""
Breast Cancer QSAR Web Application
====================================
FastAPI backend that loads trained models and predicts:
  - pIC50 (quantitative activity) via regression models
  - Activity label (Active / Moderate / Inactive) via classification models
  - Target profile: which targets the compound is active against

Run with:
  cd /Users/precision/Desktop/Breast_Cancer_ML/webapp
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os, json, base64, io, logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Draw import rdMolDraw2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
MODELS_REG = BASE / "models" / "regression"
MODELS_CLF = BASE / "models" / "classification"
RESULTS    = BASE / "results"

# ─── Constants ───────────────────────────────────────────────────────────────
ACTIVE_THR   = 7.0
MODERATE_THR = 5.0
FP_RADIUS    = 2
FP_NBITS     = 2048
RDKIT_DESC   = [
    'MolLogP', 'MolMR', 'TPSA', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
    'FractionCSP3', 'HeavyAtomCount', 'NumHeteroatoms',
    'BalabanJ', 'BertzCT'
]

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Breast Cancer QSAR Predictor", version="2.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ─── Model Cache ─────────────────────────────────────────────────────────────
REG_MODELS: dict = {}
CLF_MODELS: dict = {}
METADATA:   dict = {}
TARGETS:    list = []

def load_models():
    global REG_MODELS, CLF_MODELS, METADATA, TARGETS
    logger.info("Loading models...")

    # Load metadata
    meta_path = RESULTS / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            METADATA = json.load(f)

    # Load regression models
    for pkl in sorted(MODELS_REG.glob("*_best_reg.pkl")):
        target = pkl.stem.replace("_best_reg", "")
        try:
            REG_MODELS[target] = joblib.load(pkl)
        except Exception as e:
            logger.warning(f"Failed to load reg model {target}: {e}")

    # Load classification models
    for pkl in sorted(MODELS_CLF.glob("*_best_clf.pkl")):
        target = pkl.stem.replace("_best_clf", "")
        try:
            CLF_MODELS[target] = joblib.load(pkl)
        except Exception as e:
            logger.warning(f"Failed to load clf model {target}: {e}")

    # Targets with both models
    TARGETS = sorted(set(REG_MODELS.keys()) & set(CLF_MODELS.keys()))
    logger.info(f"Loaded {len(TARGETS)} targets with both reg+clf models")

@app.on_event("startup")
async def startup():
    load_models()

# ─── Feature Computation ─────────────────────────────────────────────────────
def compute_features(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, None
    fp  = np.array(GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS), dtype=np.float32)
    desc = []
    for d in RDKIT_DESC:
        try:
            v = getattr(Descriptors, d)(mol)
            desc.append(float(v) if v is not None else 0.0)
        except:
            desc.append(0.0)
    X = np.hstack([fp, np.array(desc, dtype=np.float32)]).reshape(1, -1)
    return X, mol

def classify_label(pic50: float) -> str:
    if pic50 >= ACTIVE_THR:
        return "Active"
    elif pic50 >= MODERATE_THR:
        return "Moderate"
    return "Inactive"

def mol_to_svg(mol, width=300, height=200) -> str:
    try:
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except:
        return ""

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class SMILESRequest(BaseModel):
    smiles: str
    name:   Optional[str] = None

class BatchRequest(BaseModel):
    smiles_list: List[str]
    names:       Optional[List[str]] = None

class PredictionResult(BaseModel):
    smiles:      str
    name:        Optional[str]
    valid:       bool
    svg:         Optional[str]
    predictions: Optional[dict]
    summary:     Optional[dict]

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "n_targets": len(TARGETS),
        "targets": TARGETS,
        "metadata": METADATA,
    })

@app.get("/targets")
async def list_targets():
    return {"targets": TARGETS, "count": len(TARGETS), "metadata": METADATA}

@app.get("/health")
async def health():
    return {"status": "ok", "reg_models": len(REG_MODELS), "clf_models": len(CLF_MODELS)}

@app.post("/predict")
async def predict_single(req: SMILESRequest):
    return _predict_one(req.smiles, req.name)

@app.post("/predict/batch")
async def predict_batch(req: BatchRequest):
    names  = req.names or [None] * len(req.smiles_list)
    results = []
    for smi, name in zip(req.smiles_list, names):
        results.append(_predict_one(smi, name))
    return {"results": results, "count": len(results)}

def _predict_one(smi: str, name: Optional[str] = None) -> dict:
    X, mol = compute_features(smi)
    if X is None or mol is None:
        return {
            "smiles": smi, "name": name, "valid": False,
            "svg": None, "predictions": None, "summary": None
        }

    svg = mol_to_svg(mol)
    predictions = {}
    active_targets    = []
    moderate_targets  = []
    inactive_targets  = []

    for target in TARGETS:
        entry = {}
        # Regression
        if target in REG_MODELS:
            try:
                pic50 = float(REG_MODELS[target].predict(X)[0])
                pic50 = round(pic50, 3)
                entry['pIC50'] = pic50
                entry['IC50_nM'] = round(10 ** (9 - pic50), 2)
            except:
                entry['pIC50'] = None
                entry['IC50_nM'] = None
        # Classification
        if target in CLF_MODELS:
            try:
                label_id = int(CLF_MODELS[target].predict(X)[0])
                label_map = {0: 'Inactive', 1: 'Moderate', 2: 'Active'}
                entry['label'] = label_map.get(label_id, 'Unknown')
                # Probability
                if hasattr(CLF_MODELS[target], 'predict_proba'):
                    proba = CLF_MODELS[target].predict_proba(X)[0]
                    classes = CLF_MODELS[target].classes_
                    prob_map = {label_map.get(int(c), 'Unknown'): round(float(p), 3)
                                for c, p in zip(classes, proba)}
                    entry['probabilities'] = prob_map
            except:
                entry['label'] = 'Unknown'

        # Combine label from regression if clf failed
        if 'label' not in entry and entry.get('pIC50') is not None:
            entry['label'] = classify_label(entry['pIC50'])

        # Target metadata
        if target in METADATA:
            entry['dataset_size']    = METADATA[target].get('n_total')
            entry['dataset_active']  = METADATA[target].get('n_active')

        predictions[target] = entry

        # Categorize
        lbl = entry.get('label', '')
        if lbl == 'Active':
            active_targets.append(target)
        elif lbl == 'Moderate':
            moderate_targets.append(target)
        elif lbl == 'Inactive':
            inactive_targets.append(target)

    # Compute physicochemical properties
    props = {}
    try:
        props = {
            'MolLogP':          round(Descriptors.MolLogP(mol), 3),
            'MW':               round(Descriptors.MolWt(mol), 2),
            'TPSA':             round(Descriptors.TPSA(mol), 2),
            'HBD':              Descriptors.NumHDonors(mol),
            'HBA':              Descriptors.NumHAcceptors(mol),
            'RotBonds':         Descriptors.NumRotatableBonds(mol),
            'Rings':            Descriptors.RingCount(mol),
            'AromaticRings':    Descriptors.NumAromaticRings(mol),
            'HeavyAtoms':       mol.GetNumHeavyAtoms(),
            'FractionCSP3':     round(Descriptors.FractionCSP3(mol), 3),
            'Lipinski_OK':      (
                Descriptors.MolWt(mol) <= 500 and
                Descriptors.MolLogP(mol) <= 5 and
                Descriptors.NumHDonors(mol) <= 5 and
                Descriptors.NumHAcceptors(mol) <= 10
            )
        }
    except:
        pass

    # Best prediction (highest pIC50)
    best_target, best_pic50 = None, -np.inf
    for t, e in predictions.items():
        p = e.get('pIC50') or -np.inf
        if p > best_pic50:
            best_pic50, best_target = p, t

    summary = {
        "n_active":    len(active_targets),
        "n_moderate":  len(moderate_targets),
        "n_inactive":  len(inactive_targets),
        "n_total":     len(TARGETS),
        "active_targets":   active_targets,
        "moderate_targets": moderate_targets,
        "inactive_targets": inactive_targets,
        "best_target": best_target,
        "best_pIC50":  round(best_pic50, 3) if best_pic50 > -np.inf else None,
        "properties":  props,
    }

    return {
        "smiles":      smi,
        "name":        name,
        "valid":       True,
        "svg":         svg,
        "predictions": predictions,
        "summary":     summary,
    }

@app.get("/results/regression")
async def get_reg_results():
    path = RESULTS / "regression_summary.csv"
    if not path.exists():
        return {"error": "Results not available yet"}
    df = pd.read_csv(path)
    return {"data": df.to_dict(orient='records')}

@app.get("/results/classification")
async def get_clf_results():
    path = RESULTS / "classification_summary.csv"
    if not path.exists():
        return {"error": "Results not available yet"}
    df = pd.read_csv(path)
    return {"data": df.to_dict(orient='records')}
