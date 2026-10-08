"""Two-phase training curriculum (v2 rewrite of ``run_stage``).

Phase 1 — Latent-space clustering
    The classification head is frozen; the three feature extractors (+ the
    fusion gate) are trained with :class:`SupervisedContrastiveLoss` on the
    fused 512-D embeddings. Validation uses leave-one-out kNN accuracy over
    val embeddings (default ``k=5``) for early stopping.

Phase 2 — Classification
    The feature extractors (+ fusion gate) are frozen; the classification
    head is trained with :class:`FocalLoss` (``gamma=2.0``) to focus on
    hard minority-class samples. Early stopping on val accuracy.

:func:`run_stage` is the direct successor of the v1 routine of the same
name (same ``(model, train_loader, val_loader, ...)`` calling convention,
now dispatching on ``phase``); :func:`train_model_two_phase` orchestrates
both phases with per-phase checkpoints, resume support and JSON logs.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.data import class_counts
from src.losses import FocalLoss, SupervisedContrastiveLoss


# ── Batch / device helpers ───────────────────────────────────────────────
def unpack_batch(batch, device):
    """Support ``(imgs, labels)`` pairs and ``(imgs, landmarks, labels)``."""
    if len(batch) == 3:
        imgs, landmarks, labels = batch
        return (imgs.to(device, non_blocking=True),
                landmarks.to(device, non_blocking=True),
                labels.to(device, non_blocking=True))
    imgs, labels = batch
    imgs = imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return imgs, None, labels


def _autocast(device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


def _scaler(device):
    return torch.amp.GradScaler("cuda") if device.type == "cuda" else nullcontext()


def compute_metrics(preds, labels):
    p, l = np.asarray(preds), np.asarray(labels)
    acc = float((p == l).mean()) * 100.0
    f1 = float(f1_score(l, p, average="weighted", zero_division=0)) * 100.0
    return acc, f1


@torch.no_grad()
def knn_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Leave-one-out kNN accuracy over L2-normalized embeddings (%)."""
    z = F.normalize(embeddings.float(), p=2, dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(-float("inf"))
    k = min(k, sim.size(0) - 1)
    _, topk = sim.topk(k, dim=1)
    votes = labels[topk]                                   # (N, k)
    preds = torch.mode(votes, dim=1).values
    return float((preds == labels).float().mean()) * 100.0


# ── Checkpoint I/O ───────────────────────────────────────────────────────
def save_ckpt(state, fname, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, fname))
    print(f"    Saved -> {fname}")


def load_ckpt(model, fname, checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {fname} not found")
        return 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    metric = ckpt.get("best_metric", ckpt.get("best_acc", 0.0))
    print(f"  Loaded {fname} — best_metric={metric:.2f}")
    return ckpt.get("epoch", 0), metric


# ── Phase 1: supervised contrastive training ─────────────────────────────
def train_one_epoch_supcon(model, loader, optimizer, scheduler, scaler,
                           criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs, landmarks, labels = unpack_batch(batch, device)
        optimizer.zero_grad()
        with _autocast(device):
            fused, _, _ = model.encode(imgs, landmarks)
            loss = criterion(fused, labels)
        if isinstance(scaler, nullcontext):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_supcon(model, loader, criterion, device, k=5):
    model.eval()
    total_loss, embs, labs = 0.0, [], []
    for batch in loader:
        imgs, landmarks, labels = unpack_batch(batch, device)
        with _autocast(device):
            fused, _, _ = model.encode(imgs, landmarks)
            total_loss += criterion(fused, labels).item()
        embs.append(fused.detach().cpu())
        labs.append(labels.detach().cpu())
    embs = torch.cat(embs)
    labs = torch.cat(labs)
    return total_loss / max(len(loader), 1), knn_accuracy(embs, labs, k=k)


# ── Phase 2: focal-loss classifier training ──────────────────────────────
def train_one_epoch_focal(model, loader, optimizer, scheduler, scaler,
                          criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        imgs, landmarks, labels = unpack_batch(batch, device)
        optimizer.zero_grad()
        with _autocast(device):
            logits, _ = model(imgs, landmarks)
            loss = criterion(logits, labels)
        if isinstance(scaler, nullcontext):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / max(len(loader), 1), acc, f1


@torch.no_grad()
def validate_classification(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, fw_log = [], [], []
    for batch in loader:
        imgs, landmarks, labels = unpack_batch(batch, device)
        with _autocast(device):
            logits, weights = model(imgs, landmarks)
            total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        fw_log.append(weights.cpu().numpy())
    fw = np.concatenate(fw_log, axis=0).mean(axis=0)
    acc, f1 = compute_metrics(all_preds, all_labels)
    return total_loss / max(len(loader), 1), acc, f1, fw


# ── Rewritten run_stage: two-phase dispatch ──────────────────────────────
def run_stage(
    model,
    train_loader,
    val_loader,
    *,
    phase: str,
    epochs: int,
    lr: float,
    ckpt_name: str,
    cfg=None,
    patience: Optional[int] = None,
    # Phase-1 options
    supcon_temperature: float = 0.07,
    knn_k: int = 5,
    # Phase-2 options
    focal_gamma: float = 2.0,
    focal_alpha=None,
    grad_clip: float = 1.0,
):
    """Train one curriculum phase (v2 rewrite of v1 ``run_stage``).

    Args:
        phase: ``"contrastive"`` (Phase 1: freeze head, SupCon on fused
            embeddings) or ``"classification"`` (Phase 2: freeze
            extractors, Focal loss on the head).
        focal_alpha: ``None`` | ``"balanced"`` (inverse-frequency weights
            from the training set) | per-class weight tensor.
    """
    from src.config import cfg as _default_cfg
    cfg = cfg or _default_cfg
    if phase not in ("contrastive", "classification"):
        raise ValueError(f"phase must be 'contrastive' or 'classification', got {phase!r}.")
    device = cfg.DEVICE
    patience = cfg.PATIENCE if patience is None else patience
    model.to(device)

    if phase == "contrastive":
        model.freeze_classifier(True)
        model.freeze_feature_extractors(False)
        criterion: nn.Module = SupervisedContrastiveLoss(supcon_temperature)
        metric_name = "knn_acc"
    else:
        model.freeze_classifier(False)
        model.freeze_feature_extractors(True)
        if isinstance(focal_alpha, str) and focal_alpha == "balanced":
            counts = class_counts(train_loader.dataset).float()
            focal_alpha = (counts.sum() / (len(counts) * counts.clamp_min(1))).to(device)
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        metric_name = "val_acc"

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.3, anneal_strategy="cos", div_factor=10, final_div_factor=100)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else nullcontext()

    best_metric, patience_ct, log = 0.0, 0, []
    print(f"\n{'=' * 65}")
    print(f"  Phase '{phase}' — {epochs} epochs | max_lr={lr:.2e} | "
          f"trainable={sum(p.numel() for p in trainable):,}")
    print(f"  Early stopping on {metric_name} (patience={patience})")
    print(f"{'=' * 65}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if phase == "contrastive":
            tr_loss = train_one_epoch_supcon(
                model, train_loader, optimizer, scheduler, scaler, criterion,
                device, grad_clip)
            vl_loss, vl_metric = validate_supcon(
                model, val_loader, criterion, device, k=knn_k)
            print(f"  E{epoch:02d}/{epochs} | tr_supcon={tr_loss:.4f} | "
                  f"vl_supcon={vl_loss:.4f} knn={vl_metric:.1f}% | {time.time()-t0:.0f}s")
            row = {"epoch": epoch, "phase": phase, "tr_loss": tr_loss,
                   "vl_loss": vl_loss, "knn_acc": vl_metric,
                   "elapsed_sec": time.time() - t0}
        else:
            tr_loss, tr_acc, tr_f1 = train_one_epoch_focal(
                model, train_loader, optimizer, scheduler, scaler, criterion,
                device, grad_clip)
            vl_loss, vl_acc, vl_f1, fw = validate_classification(
                model, val_loader, criterion, device)
            vl_metric = vl_acc
            fw_str = ",".join(f"{w:.2f}" for w in fw)
            print(f"  E{epoch:02d}/{epochs} | tr_acc={tr_acc:.1f}% f1={tr_f1:.1f}% | "
                  f"val_acc={vl_acc:.1f}% f1={vl_f1:.1f}% | fw=[{fw_str}] | "
                  f"{time.time()-t0:.0f}s")
            row = {"epoch": epoch, "phase": phase, "tr_loss": tr_loss,
                   "vl_loss": vl_loss, "tr_acc": tr_acc, "vl_acc": vl_acc,
                   "vl_f1": vl_f1, "elapsed_sec": time.time() - t0}
        log.append(row)

        if vl_metric > best_metric:
            best_metric, patience_ct = vl_metric, 0
            save_ckpt({"epoch": epoch, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(),
                       "best_metric": best_metric, "phase": phase},
                      ckpt_name, cfg.CHECKPOINT_DIR)
        else:
            patience_ct += 1
            if patience_ct >= patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\n  Phase '{phase}' best {metric_name}: {best_metric:.2f}")
    return log, best_metric


# ── Orchestrator ─────────────────────────────────────────────────────────
def train_model_two_phase(
    model,
    loaders: Dict[str, torch.utils.data.DataLoader],
    cfg=None,
    ckpt_prefix: str = "TriBranch",
    phase_epochs: Optional[Tuple[int, int]] = None,
    phase_lrs: Optional[Tuple[float, float]] = None,
    resume_phase: int = 1,
    on_phase_done=None,
):
    """Run Phase 1 (SupCon) → Phase 2 (Focal) for the tri-branch model.

    Args:
        loaders: dict from :func:`src.data.get_dataloaders` with
            ``train_contrast`` / ``train_focal`` / ``val`` entries.
        resume_phase: 1 or 2 — earlier phases are skipped and their
            checkpoints reloaded (for resuming after pre-emption).
        on_phase_done: optional callback ``(phase_num, cfg)`` fired after
            each phase checkpoint lands on disk (hook for persist/zip).
    Returns:
        ``(summary, ckpt_path, full_log)``.
    """
    from src.config import cfg as _default_cfg
    cfg = cfg or _default_cfg
    e1, e2 = phase_epochs or (cfg.PHASE1_EPOCHS, cfg.PHASE2_EPOCHS)
    lr1, lr2 = phase_lrs or (cfg.PHASE1_LR, cfg.PHASE2_LR)

    print(f"\n{'#' * 65}\n#  TWO-PHASE TRAINING: {ckpt_prefix}\n{'#' * 65}")
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_p:,}")

    full_log: List[dict] = []
    best1 = best2 = float("nan")

    # ── Phase 1: latent-space clustering ────────────────────────────────
    if resume_phase <= 1:
        log1, best1 = run_stage(
            model, loaders["train_contrast"], loaders["val"],
            phase="contrastive", epochs=e1, lr=lr1,
            ckpt_name=f"{ckpt_prefix}_phase1_best.pth", cfg=cfg,
            supcon_temperature=cfg.SUPCON_TEMPERATURE, knn_k=cfg.PHASE1_KNN_K)
        full_log += log1
    else:
        print("  SKIP Phase 1 (already completed) — loading checkpoint.")
    load_ckpt(model, f"{ckpt_prefix}_phase1_best.pth", cfg.CHECKPOINT_DIR, cfg.DEVICE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if on_phase_done:
        on_phase_done(1, cfg)

    # ── Phase 2: focal classification ───────────────────────────────────
    if resume_phase <= 2:
        log2, best2 = run_stage(
            model, loaders["train_focal"], loaders["val"],
            phase="classification", epochs=e2, lr=lr2,
            ckpt_name=f"{ckpt_prefix}_phase2_best.pth", cfg=cfg,
            focal_gamma=cfg.FOCAL_GAMMA, focal_alpha=cfg.FOCAL_ALPHA)
        full_log += log2
    else:
        load_ckpt(model, f"{ckpt_prefix}_phase2_best.pth", cfg.CHECKPOINT_DIR, cfg.DEVICE)
    if on_phase_done:
        on_phase_done(2, cfg)

    log_path = os.path.join(cfg.LOG_DIR, f"{ckpt_prefix}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2)

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ckpt_prefix}_phase2_best.pth")
    print(f"\n{'=' * 65}\n  {ckpt_prefix} TRAINING COMPLETE\n"
          f"  Phase 1 best kNN: {best1:.2f}  |  Phase 2 best acc: {best2:.2f}\n"
          f"  Checkpoint -> {ckpt_path}\n{'=' * 65}")
    summary = {"phase1_knn": best1, "phase2_acc": best2, "ckpt_path": ckpt_path}
    return summary, ckpt_path, full_log
