"""Training loop with validation-only model selection."""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels, all_probabilities = [], []

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1).float()

        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach()) * batch.num_graphs
        all_labels.append(targets.detach().cpu())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu())

    return (
        total_loss / len(loader.dataset),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probabilities).numpy(),
    )


def train_model(model, train_loader, val_loader, device, epochs, lr, pos_weight=None,
                patience=None, checkpoint_path=None):
    model = model.to(device)

    weight = None if pos_weight is None else torch.tensor(pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best = {"score": -1.0, "epoch": -1, "state": None}

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, y_val, p_val = run_epoch(model, val_loader, device, criterion)
        val_roc_auc = roc_auc_score(y_val, p_val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_roc_auc,
        })

        if val_roc_auc > best["score"]:
            best = {
                "score": val_roc_auc,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }

        if patience is not None and epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])

    if checkpoint_path is not None:
        torch.save(
            {
                "model_state": best["state"],
                "history": history,
                "best_epoch": best["epoch"],
                "best_val_roc_auc": best["score"],
            },
            checkpoint_path,
        )

    return model, pd.DataFrame(history)


def load_run(path, config):
    # returns None unless a checkpoint exists that was written with exactly this config,
    # so changing the model or the budget silently retrains instead of loading stale weights
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("config") != config:
        return None
    return saved["model_state"], pd.DataFrame(saved["history"])


def save_run(path, model, history, config):
    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history.to_dict("records"),
            "config": config,
        },
        path,
    )


def predict(model, loader, device):
    model = model.to(device)
    model.eval()

    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            all_labels.append(batch.y.view(-1).float().cpu())
            all_probabilities.append(torch.sigmoid(model(batch)).cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_probabilities).numpy()
