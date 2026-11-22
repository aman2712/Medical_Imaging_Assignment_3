"""Morgan/ECFP fingerprints, the logistic-regression baseline and Tanimoto similarity."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.evaluation import select_threshold


def morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        fingerprints[i] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
    return fingerprints


def train_baseline(x_train, y_train, c=1.0, class_weight=None, seed=42):
    model = LogisticRegression(
        C=c,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def select_baseline(x_train, y_train, x_val, y_val, grid, seed=42):
    rows = []
    best = None

    for c, class_weight in grid:
        model = train_baseline(x_train, y_train, c=c, class_weight=class_weight, seed=seed)
        probabilities = model.predict_proba(x_val)[:, 1]
        score = roc_auc_score(y_val, probabilities)

        rows.append({
            "C": c,
            "class_weight": str(class_weight),
            "val_roc_auc": round(score, 4),
        })
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "C": c,
                "class_weight": class_weight,
                "model": model,
                "probabilities": probabilities,
            }

    settings = {
        "C": best["C"],
        "class_weight": best["class_weight"],
        "val_roc_auc": round(best["score"], 4),
        "threshold": select_threshold(y_val, best["probabilities"]),
    }
    return best["model"], settings, pd.DataFrame(rows)


def tanimoto(fingerprint, fingerprint_matrix):
    # cast out of uint8 first: an intersection can reach 2048 and would wrap round
    query = fingerprint.astype(np.float64)
    matrix = fingerprint_matrix.astype(np.float64)

    intersection = matrix @ query
    union = query.sum() + matrix.sum(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros(len(matrix)), where=union > 0)


def nearest_training_molecule(query_fingerprint, train_fingerprints):
    similarities = tanimoto(query_fingerprint, train_fingerprints)
    best = int(np.argmax(similarities))
    return best, float(similarities[best])
