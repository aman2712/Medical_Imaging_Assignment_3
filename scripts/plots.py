"""Figures: molecule grids, graph views, learning curves, ROC/PR curves, confusion matrices."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

POSITIVE_COLOUR = "#0072B2"
NEGATIVE_COLOUR = "#D55E00"


def draw_molecules(smiles_list, labels, titles=None, columns=3):
    """Draw a grid of chemical structures with the SMILES string and label."""
    rows = -(-len(smiles_list) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.6 * rows))
    flat = axes.ravel()

    for i, (ax, smiles, label) in enumerate(zip(flat, smiles_list, labels)):
        molecule = Chem.MolFromSmiles(smiles)
        ax.imshow(Draw.MolToImage(molecule, size=(400, 280)))
        ax.set_title(titles[i] if titles is not None else f"label = {int(label)}", fontsize=10)
        ax.set_xlabel(textwrap.fill(smiles, 36), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in flat[len(smiles_list):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def draw_graph_view(graph):
    molecule = Chem.MolFromSmiles(graph.smiles)
    rdDepictor.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    positions = {
        atom.GetIdx(): (
            conformer.GetAtomPosition(atom.GetIdx()).x,
            conformer.GetAtomPosition(atom.GetIdx()).y,
        )
        for atom in molecule.GetAtoms()
    }

    for atom in molecule.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(Draw.MolToImage(molecule, size=(520, 420)))
    axes[0].set_title("Chemical structure")
    axes[0].axis("off")

    ax = axes[1]
    for source, target in graph.edge_index.t().tolist():
        if source < target:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            ax.plot([x0, x1], [y0, y1], color="#AAAAAA", linewidth=1.5, zorder=1)

    ax.scatter(
        [p[0] for p in positions.values()],
        [p[1] for p in positions.values()],
        s=430, color=POSITIVE_COLOUR, zorder=2,
    )
    for index, (x, y) in positions.items():
        ax.text(x, y, str(index), ha="center", va="center", color="white", fontsize=9, zorder=3)

    ax.set_title("Node-edge graph")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_history(history, title=None):
    best = history["val_roc_auc"].idxmax()
    best_epoch = int(history.loc[best, "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].plot(history["epoch"], history["train_loss"], color=POSITIVE_COLOUR, label="train")
    axes[0].plot(history["epoch"], history["val_loss"], color=NEGATIVE_COLOUR, label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend(frameon=False)

    axes[1].plot(history["epoch"], history["val_roc_auc"], color=POSITIVE_COLOUR)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation ROC-AUC")

    for ax in axes:
        ax.axvline(best_epoch, linestyle="--", linewidth=1, color="#888888")
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].annotate(
        f"best epoch {best_epoch}",
        xy=(best_epoch, history.loc[best, "val_roc_auc"]),
        xytext=(4, -12), textcoords="offset points", fontsize=8, color="#555555",
    )

    if title is not None:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_roc_pr(curves, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    for label, (y_true, probabilities) in curves.items():
        false_positive, true_positive, _ = roc_curve(y_true, probabilities)
        axes[0].plot(false_positive, true_positive,
                     label=f"{label} ({roc_auc_score(y_true, probabilities):.3f})")

        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        axes[1].plot(recall, precision,
                     label=f"{label} ({average_precision_score(y_true, probabilities):.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle=":", linewidth=1, color="#999999")
    first = next(iter(curves.values()))[0]
    axes[1].axhline(float(np.mean(first)), linestyle=":", linewidth=1, color="#999999")

    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC", fontsize=10)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall", fontsize=10)

    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, ax=None, title=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax.imshow(matrix, cmap="Blues")

    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center", fontsize=11,
                color="white" if value > matrix.max() / 2 else "black")

    ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
    ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
    if title is not None:
        ax.set_title(title, fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_similar_pair(query_smiles, neighbour_smiles, info):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))

    for ax, smiles, caption in zip(
        axes, [query_smiles, neighbour_smiles], [info["query"], info["neighbour"]]
    ):
        ax.imshow(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(420, 320)))
        ax.set_title(caption, fontsize=9)
        ax.set_xlabel(textwrap.fill(smiles, 40), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Tanimoto similarity {info['similarity']:.3f}", fontsize=11)
    plt.tight_layout()
    plt.show()
