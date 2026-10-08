"""Objectives for the two-phase curriculum.

Phase 1 — :class:`SupervisedContrastiveLoss` (SupCon, Khosla et al., 2020)
    Clusters the fused 512-D embeddings by class so overlapping compound
    pairs (e.g. *Sadly Angry* vs *Sadly Disgusted*) are pushed apart before
    any classifier is trained.

Phase 2 — :class:`FocalLoss` (Lin et al., 2017, ``gamma=2.0``)
    Trains the classifier head while dynamically down-weighting easy
    majority-class samples and up-weighting hard minority-class samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss over L2-normalized embeddings.

    For every anchor, all same-class samples in the batch act as positives
    and every other sample as a negative (self-contrast excluded).

    Args:
        temperature: contrast temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.size(0) < 2:
            return features.sum() * 0.0  # degenerate batch → zero loss w/ grad graph
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature  # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()  # stability

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T)                                  # (B, B)
        self_mask = torch.eye(features.size(0), dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~self_mask                                 # drop self

        exp_logits = torch.exp(logits) * (~self_mask)                    # denom w/o self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        n_pos = pos_mask.sum(dim=1)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / n_pos.clamp_min(1)
        valid = n_pos > 0  # anchors with no positives contribute nothing
        if valid.sum() == 0:
            return features.sum() * 0.0
        return -(mean_log_prob_pos * valid.float()).sum() / valid.float().sum().clamp_min(1)


class FocalLoss(nn.Module):
    """Multi-class focal loss: ``FL = -α (1 − p_t)^γ log(p_t)``.

    Args:
        gamma: focusing parameter (default 2.0 per the v2 spec).
        alpha: optional per-class weights — ``None``, a ``(C,)`` tensor, or a
            scalar applied uniformly. Use ``"balanced"`` handling in
            :mod:`src.train` to derive inverse-frequency weights.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float)
        self.register_buffer("alpha", alpha)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction {reduction!r}.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)                       # (B, C)
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)    # (B,)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[targets]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
