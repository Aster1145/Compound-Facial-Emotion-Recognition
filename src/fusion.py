"""Dynamic gated fusion (AttentionFusion), retained from v1.

The gate consumes the concatenated branch embeddings and emits dynamic
``α, β, γ`` weights (softmax over branches) for the final fused embedding::

    [f_local ‖ f_global ‖ f_geom] → Linear → ReLU → Linear → Softmax(α,β,γ)
    fused = α·f_local + β·f_global + γ·f_geom
"""
import torch
import torch.nn as nn


class GenericAttentionFusion(nn.Module):
    """N-branch gated softmax fusion (identical math to v1)."""

    def __init__(self, embed_dim=512, num_branches=3):
        super().__init__()
        self.num_branches = num_branches
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_branches, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_branches),
            nn.Softmax(dim=-1),
        )

    def forward(self, feats):
        # feats: list of (B, embed_dim) tensors, len == num_branches
        combined = torch.cat(feats, dim=-1)
        weights = self.gate(combined)                      # (B, num_branches)
        fused = sum(weights[:, i:i + 1] * feats[i] for i in range(self.num_branches))
        return fused, weights


class AttentionFusion(GenericAttentionFusion):
    """3-branch fusion gate producing the dynamic α, β, γ weights.

    Accepts either ``forward(f_local, f_global, f_geom)`` or
    ``forward([f_local, f_global, f_geom])``. Returns ``(fused, weights)``
    with ``weights`` columns ordered ``[α, β, γ]``.
    """

    def __init__(self, embed_dim=512):
        super().__init__(embed_dim=embed_dim, num_branches=3)

    def forward(self, *feats):  # type: ignore[override]
        if len(feats) == 1 and isinstance(feats[0], (list, tuple)):
            feats = tuple(feats[0])
        assert len(feats) == 3, f"AttentionFusion expects 3 branches, got {len(feats)}."
        return super().forward(list(feats))
