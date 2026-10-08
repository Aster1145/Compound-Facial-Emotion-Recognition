"""Evaluation metrics for the tri-branch model (handles triple batches)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.train import unpack_batch


@torch.no_grad()
def evaluate_tribranch(model, loader, cfg=None, class_names=None):
    """Evaluate an 11-way tri-branch model on a compound-labeled loader.

    Returns a summary dict with accuracy, macro/weighted precision, recall,
    F1, ROC-AUC (OvR, macro), the confusion matrix, the sklearn report dict
    and the mean fusion weights ``[α, β, γ]``.
    """
    from src.config import cfg as _default_cfg
    cfg = cfg or _default_cfg
    names = class_names or cfg.COMPOUND_CLASSES
    device = cfg.DEVICE
    model.to(device).eval()

    all_probs, all_preds, all_labels, fw_log = [], [], [], []
    for batch in loader:
        imgs, landmarks, labels = unpack_batch(batch, device)
        logits, weights = model(imgs, landmarks)
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        fw_log.append(weights.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100.0,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0,
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0,
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) * 100.0,
        "auc_macro": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, target_names=names,
                                        zero_division=0, output_dict=True),
        "fusion_weights_mean": np.concatenate(fw_log, axis=0).mean(axis=0).tolist(),
        "n_samples": len(y_true),
    }
