"""Stratified random and Bemis-Murcko scaffold splits, with verification helpers."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def murcko_scaffold(smiles, include_chirality=False):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=include_chirality
    )


def random_split(labels, fractions=(0.70, 0.15, 0.15), seed=42):
    labels = np.asarray(labels)
    positions = np.arange(len(labels))

    train, rest = train_test_split(
        positions, train_size=fractions[0], random_state=seed, stratify=labels
    )
    validation_share = fractions[1] / (fractions[1] + fractions[2])
    validation, test = train_test_split(
        rest, train_size=validation_share, random_state=seed, stratify=labels[rest]
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def scaffold_split(smiles_list, fractions=(0.70, 0.15, 0.15), seed=42):
    groups = {}
    for position, smiles in enumerate(smiles_list):
        groups.setdefault(murcko_scaffold(smiles), []).append(position)

    rng = np.random.default_rng(seed)
    keys = list(groups)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    keys.sort(key=lambda key: len(groups[key]), reverse=True)

    n_total = len(smiles_list)
    n_train = int(fractions[0] * n_total)
    n_validation = int(fractions[1] * n_total)

    train, validation, test = [], [], []
    for key in keys:
        group = groups[key]
        if len(train) + len(group) <= n_train:
            train += group
        elif len(validation) + len(group) <= n_validation:
            validation += group
        else:
            test += group

    return np.sort(train), np.sort(validation), np.sort(test)


def verify_split(split, n_total, canonical_smiles, scaffolds=None):
    train, validation, test = split
    subsets = {"train": train, "validation": validation, "test": test}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]

    canonical = np.asarray(canonical_smiles, dtype=object)
    covered = np.concatenate([train, validation, test])

    checks = {
        "molecules assigned": len(covered),
        "distinct molecules covered": len(np.unique(covered)),
        "indices shared between subsets": sum(
            len(set(subsets[a]) & set(subsets[b])) for a, b in pairs
        ),
        "canonical SMILES shared between subsets": sum(
            len(set(canonical[subsets[a]]) & set(canonical[subsets[b]])) for a, b in pairs
        ),
    }
    if scaffolds is not None:
        scaffold_array = np.asarray(scaffolds, dtype=object)
        checks["scaffolds shared between subsets"] = sum(
            len(set(scaffold_array[subsets[a]]) & set(scaffold_array[subsets[b]]))
            for a, b in pairs
        )

    assert checks["molecules assigned"] == n_total
    assert checks["distinct molecules covered"] == n_total
    assert checks["indices shared between subsets"] == 0
    assert checks["canonical SMILES shared between subsets"] == 0
    if scaffolds is not None:
        assert checks["scaffolds shared between subsets"] == 0

    return pd.Series(checks)


def split_summary(split, labels, scaffolds=None):
    labels = np.asarray(labels)
    if scaffolds is not None:
        scaffolds = np.asarray(scaffolds, dtype=object)

    rows = {}
    for name, indices in zip(("train", "validation", "test"), split):
        row = {
            "molecules": len(indices),
            "positive": int(labels[indices].sum()),
            "negative": int((1 - labels[indices]).sum()),
            "positive_proportion": round(float(labels[indices].mean()), 4),
        }
        if scaffolds is not None:
            row["unique_scaffolds"] = len(set(scaffolds[indices]))
        rows[name] = row

    return pd.DataFrame(rows).T
