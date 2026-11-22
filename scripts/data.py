import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def graph_is_valid(graph):
    return bool(torch.isfinite(graph.y).all()) and graph.x.shape[0] > 0


def inspect_dataset(dataset):
    rows = []

    for i, graph in enumerate(dataset):
        label = graph.y.view(-1)
        rows.append({
            "index": i,
            "smiles": graph.smiles,
            "canonical_smiles": canonical_smiles(graph.smiles),
            "label": float(label[0]) if torch.isfinite(label).all() else np.nan,
            "n_atoms": int(graph.x.shape[0]),
            "n_bonds": int(graph.edge_index.shape[1] // 2),
            "finite_label": bool(torch.isfinite(label).all()),
            "has_atoms": bool(graph.x.shape[0] > 0),
        })

    table = pd.DataFrame(rows).set_index("index")
    table["valid_smiles"] = table["canonical_smiles"].notna()
    return table


def find_duplicates(table):
    valid = table[table["valid_smiles"]]
    groups = valid.groupby("canonical_smiles")["label"]

    summary = pd.DataFrame({
        "n_records": groups.size(),
        "n_distinct_labels": groups.nunique(),
    })
    duplicated = summary[summary["n_records"] > 1]

    consistent = duplicated[duplicated["n_distinct_labels"] == 1]
    conflicting = duplicated[duplicated["n_distinct_labels"] > 1]
    return consistent, conflicting


def clean_dataset(table):
    steps = []

    valid = table[table["valid_smiles"] & table["has_atoms"] & table["finite_label"]]
    steps.append(("invalid SMILES or empty graphs", len(table) - len(valid)))

    _, conflicting = find_duplicates(valid)

    resolved = valid[~valid["canonical_smiles"].isin(conflicting.index)]
    steps.append(("records in label-conflicting groups", len(valid) - len(resolved)))

    deduplicated = resolved[~resolved["canonical_smiles"].duplicated(keep="first")]
    steps.append(("extra copies of consistent duplicates", len(resolved) - len(deduplicated)))

    report = pd.DataFrame(steps, columns=["step", "records_removed"])
    return deduplicated.index.to_numpy(), report
