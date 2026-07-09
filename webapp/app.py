#!/usr/bin/env python3
"""
BreastCAR — Multi-Target QSAR Web Application
FastAPI backend: downloads trained models from GitHub Release on first startup,
then loads them lazily (per-target on demand) to minimise memory usage.
"""

import os, json, tarfile, threading, base64, io, logging
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Draw import rdMolDraw2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
# On Render, use /opt/models for persistent disk; fall back to repo root
MODEL_ROOT = Path(os.environ.get("MODEL_DIR", str(BASE / "models")))
MODELS_REG = MODEL_ROOT / "regression"
MODELS_CLF = MODEL_ROOT / "classification"

# ─── GitHub Release download URLs ─────────────────────────────────────────────
RELEASE_BASE = "https://github.com/precisionmatics/BreastCAR-QSAR/releases/download/v1.0"
ASSETS = {
    "models_reg.tar.gz": f"{RELEASE_BASE}/models_reg.tar.gz",
    "models_clf.tar.gz": f"{RELEASE_BASE}/models_clf.tar.gz",
}

# ─── Constants ────────────────────────────────────────────────────────────────
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

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="BreastCAR QSAR Predictor", version="2.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

TARGETS: list = []
METADATA: dict = {}
_model_lock = threading.Lock()
_reg_cache: dict = {}
_clf_cache: dict = {}

# ─── Model Download ───────────────────────────────────────────────────────────
def download_models():
    """Download and extract model archives from GitHub Release if not present."""
    MODELS_REG.mkdir(parents=True, exist_ok=True)
    MODELS_CLF.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    reg_files = list(MODELS_REG.glob("*_best_reg.pkl"))
    clf_files = list(MODELS_CLF.glob("*_best_clf.pkl"))
    if len(reg_files) >= 50 and len(clf_files) >= 50:
        logger.info(f"Models already present: {len(reg_files)} reg, {len(clf_files)} clf")
        return

    for fname, url in ASSETS.items():
        dest = MODEL_ROOT / fname
        if dest.exists():
            logger.info(f"Archive already exists: {fname}")
        else:
            logger.info(f"Downloading {fname} from GitHub Release (~160-170MB)...")
            try:
                urllib.request.urlretrieve(url, dest)
                logger.info(f"Downloaded {fname} ({dest.stat().st_size // 1024 // 1024}MB)")
            except Exception as e:
                logger.error(f"Failed to download {fname}: {e}")
                continue

        logger.info(f"Extracting {fname}...")
        try:
            with tarfile.open(dest, 'r:gz') as tf:
                tf.extractall(MODEL_ROOT)
            logger.info(f"Extracted {fname}")
        except Exception as e:
            logger.error(f"Failed to extract {fname}: {e}")

# ─── Lazy Model Loading ───────────────────────────────────────────────────────
def get_reg_model(target: str):
    if target not in _reg_cache:
        with _model_lock:
            if target not in _reg_cache:
                pkl = MODELS_REG / f"{target}_best_reg.pkl"
                if pkl.exists():
                    _reg_cache[target] = joblib.load(pkl)
                else:
                    return None
    return _reg_cache.get(target)

def get_clf_model(target: str):
    if target not in _clf_cache:
        with _model_lock:
            if target not in _clf_cache:
                pkl = MODELS_CLF / f"{target}_best_clf.pkl"
                if pkl.exists():
                    _clf_cache[target] = joblib.load(pkl)
                else:
                    return None
    return _clf_cache.get(target)

def discover_targets():
    global TARGETS
    reg_targets = {p.stem.replace("_best_reg", "") for p in MODELS_REG.glob("*_best_reg.pkl")
                   if not p.name.startswith("._")}
    clf_targets = {p.stem.replace("_best_clf", "") for p in MODELS_CLF.glob("*_best_clf.pkl")
                   if not p.name.startswith("._")}
    TARGETS = sorted(reg_targets & clf_targets)
    logger.info(f"Discovered {len(TARGETS)} targets with both reg+clf models")

@app.on_event("startup")
async def startup():
    download_models()
    discover_targets()

# ─── Feature Computation ──────────────────────────────────────────────────────
def compute_features(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, None
    fp   = np.array(GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS), dtype=np.float32)
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
    if pic50 >= ACTIVE_THR:    return "Active"
    if pic50 >= MODERATE_THR:  return "Moderate"
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

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class SMILESRequest(BaseModel):
    smiles: str
    name:   Optional[str] = None

class BatchRequest(BaseModel):
    smiles_list: List[str]
    names:       Optional[List[str]] = None

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":  request,
        "n_targets": len(TARGETS),
        "targets":  TARGETS,
        "metadata": METADATA,
    })

@app.get("/targets")
async def list_targets():
    return {"targets": TARGETS, "count": len(TARGETS)}

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "n_targets":    len(TARGETS),
        "reg_cached":   len(_reg_cache),
        "clf_cached":   len(_clf_cache),
    }

@app.post("/predict")
async def predict_single(req: SMILESRequest):
    return _predict_one(req.smiles, req.name)

@app.post("/predict/batch")
async def predict_batch(req: BatchRequest):
    names   = req.names or [None] * len(req.smiles_list)
    results = [_predict_one(smi, n) for smi, n in zip(req.smiles_list, names)]
    return {"results": results, "count": len(results)}

def _predict_one(smi: str, name: Optional[str] = None) -> dict:
    X, mol = compute_features(smi)
    if X is None:
        return {"smiles": smi, "name": name, "valid": False,
                "svg": None, "predictions": None, "summary": None}

    svg = mol_to_svg(mol)
    predictions   = {}
    active_tgts   = []
    moderate_tgts = []
    inactive_tgts = []

    for target in TARGETS:
        entry = {}
        reg = get_reg_model(target)
        if reg is not None:
            try:
                pic50 = round(float(reg.predict(X)[0]), 3)
                entry['pIC50']   = pic50
                entry['IC50_nM'] = round(10 ** (9 - pic50), 2)
            except:
                entry['pIC50'] = None

        clf = get_clf_model(target)
        if clf is not None:
            try:
                raw = clf.predict(X)[0]
                # models may return string labels or integer-encoded labels
                if isinstance(raw, (int, np.integer)):
                    int_map = {0: 'Inactive', 1: 'Moderate', 2: 'Active'}
                    entry['label'] = int_map.get(int(raw), str(raw))
                else:
                    entry['label'] = str(raw)
            except:
                entry['label'] = 'Unknown'
            try:
                if hasattr(clf, 'predict_proba'):
                    proba  = clf.predict_proba(X)[0]
                    int_map = {0: 'Inactive', 1: 'Moderate', 2: 'Active'}
                    entry['probabilities'] = {}
                    for c, p in zip(clf.classes_, proba):
                        lbl = int_map.get(int(c), str(c)) if isinstance(c, (int, np.integer)) else str(c)
                        entry['probabilities'][lbl] = round(float(p), 3)
            except:
                pass

        if 'label' not in entry and entry.get('pIC50') is not None:
            entry['label'] = classify_label(entry['pIC50'])

        predictions[target] = entry
        lbl = entry.get('label', '')
        if lbl == 'Active':        active_tgts.append(target)
        elif lbl == 'Moderate':    moderate_tgts.append(target)
        elif lbl == 'Inactive':    inactive_tgts.append(target)

    # Physicochemical properties
    props = {}
    try:
        props = {
            'MolLogP':       round(Descriptors.MolLogP(mol), 3),
            'MW':            round(Descriptors.MolWt(mol), 2),
            'TPSA':          round(Descriptors.TPSA(mol), 2),
            'HBD':           Descriptors.NumHDonors(mol),
            'HBA':           Descriptors.NumHAcceptors(mol),
            'RotBonds':      Descriptors.NumRotatableBonds(mol),
            'Rings':         Descriptors.RingCount(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'HeavyAtoms':    mol.GetNumHeavyAtoms(),
            'FractionCSP3':  round(Descriptors.FractionCSP3(mol), 3),
            'Lipinski_OK':   (Descriptors.MolWt(mol) <= 500 and
                              Descriptors.MolLogP(mol) <= 5 and
                              Descriptors.NumHDonors(mol) <= 5 and
                              Descriptors.NumHAcceptors(mol) <= 10),
        }
    except:
        pass

    best_target, best_pic50 = None, -np.inf
    for t, e in predictions.items():
        p = e.get('pIC50') or -np.inf
        if p > best_pic50:
            best_pic50, best_target = p, t

    return {
        "smiles":      smi,
        "name":        name,
        "valid":       True,
        "svg":         svg,
        "predictions": predictions,
        "summary": {
            "n_active":         len(active_tgts),
            "n_moderate":       len(moderate_tgts),
            "n_inactive":       len(inactive_tgts),
            "n_total":          len(TARGETS),
            "active_targets":   active_tgts,
            "moderate_targets": moderate_tgts,
            "inactive_targets": inactive_tgts,
            "best_target":      best_target,
            "best_pIC50":       round(best_pic50, 3) if best_pic50 > -np.inf else None,
            "properties":       props,
        },
    }

@app.get("/results/regression")
async def get_reg_results():
    path = BASE / "revised_results" / "regression_summary.csv"
    if not path.exists():
        return {"error": "Results not available"}
    return {"data": pd.read_csv(path).to_dict(orient='records')}

@app.get("/results/classification")
async def get_clf_results():
    path = BASE / "revised_results" / "classification_summary.csv"
    if not path.exists():
        return {"error": "Results not available"}
    return {"data": pd.read_csv(path).to_dict(orient='records')}
