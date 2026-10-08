"""Dual-domain tri-branch :class:`CompoundEmotionModel` (v2).

Pipeline::

    face image (224×224) ─┬─► Branch 1 local textures (timm CNN + CBAM) ─► 512-D
                           ├─► Branch 2 global context (face ViT [CLS])   ─► 512-D
    landmarks (468×3) ─────┴─► Branch 3 facial geometry (MLP) ─────────────► 512-D
                                              │
                                     [ AttentionFusion α,β,γ ]
                                              │
                                      fused embedding (512-D)
                                              │
                                      [ MLP classifier head ]
                                              │
                                       11 compound classes

``forward(images, landmarks)`` returns ``(logits, weights)`` — the same
tuple signature as v1 — while :meth:`encode` additionally exposes the fused
embedding used by the Phase-1 contrastive objective.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.branches import GeometryMLP, GlobalContextEncoder, LocalTextureEncoder
from src.fusion import AttentionFusion


def _make_classifier(embed_dim, num_classes, dropout):
    clf = nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.5),
        nn.Linear(256, num_classes),
    )
    for m in clf.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return clf


class CompoundEmotionModel(nn.Module):
    """Dual-domain tri-branch compound-emotion classifier."""

    def __init__(
        self,
        num_classes: int = 11,
        embed_dim: int = 512,
        dropout: float = 0.3,
        local_backbone: str = "efficientnet_b0",
        vggface2_weights: Optional[str] = None,
        hf_face_vit_id: str = "trpakov/vit-face-expression",
        num_landmarks: int = 468,
        landmark_dim: int = 3,
        geometry_hidden: int = 1024,
    ):
        super().__init__()
        self.branch_names = ["local", "global", "geometry"]
        self.branches = nn.ModuleDict(
            OrderedDict(
                [
                    (
                        "local",
                        LocalTextureEncoder(
                            backbone_name=local_backbone,
                            vggface2_weights=vggface2_weights,
                            embed_dim=embed_dim,
                            dropout=dropout,
                        ),
                    ),
                    (
                        "global",
                        GlobalContextEncoder(
                            hf_model_id=hf_face_vit_id,
                            embed_dim=embed_dim,
                            dropout=dropout,
                        ),
                    ),
                    (
                        "geometry",
                        GeometryMLP(
                            num_landmarks=num_landmarks,
                            landmark_dim=landmark_dim,
                            hidden_dim=geometry_hidden,
                            embed_dim=embed_dim,
                            dropout=dropout,
                        ),
                    ),
                ]
            )
        )
        self.fusion = AttentionFusion(embed_dim=embed_dim)
        self.classifier = _make_classifier(embed_dim, num_classes, dropout)
        self._landmark_len = num_landmarks * landmark_dim

    # -- feature-extractor / head freezing for the two-phase curriculum ----
    def freeze_classifier(self, freeze: bool = True):
        """Freeze (Phase 1) or unfreeze (Phase 2) the classification head."""
        for p in self.classifier.parameters():
            p.requires_grad = not freeze

    def freeze_feature_extractors(self, freeze: bool = True):
        """Freeze (Phase 2) or unfreeze (Phase 1) branches + fusion gate."""
        for name in self.branch_names:
            self.branches[name].set_trainable(not freeze)
        for p in self.fusion.parameters():
            p.requires_grad = not freeze

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # legacy staged-curriculum hooks (kept for backward compatibility) -----
    def stage1_setup(self):
        for branch in self.branches.values():
            if hasattr(branch, "_freeze_all"):
                branch._freeze_all()
        print(f"  Stage 1 trainable params: {self.trainable_params():,}")

    def stage2_setup(self, n=8):
        for branch in self.branches.values():
            if hasattr(branch, "unfreeze_last_n_blocks"):
                branch.unfreeze_last_n_blocks(n)
        print(f"  Stage 2 trainable params: {self.trainable_params():,}")

    def stage3_setup(self):
        for branch in self.branches.values():
            if hasattr(branch, "unfreeze_all"):
                branch.unfreeze_all()
        print(f"  Stage 3 trainable params: {self.trainable_params():,}")

    # -- forward -----------------------------------------------------------
    def _as_landmarks(self, landmarks, batch_size, device, dtype):
        if landmarks is None:  # image-only fallback (geometry branch sees zeros)
            return torch.zeros(
                batch_size, self._landmark_len, device=device, dtype=dtype
            )
        return landmarks.to(device=device, dtype=dtype)

    def encode(self, images: torch.Tensor, landmarks: Optional[torch.Tensor] = None):
        """Return ``(fused, weights, branch_feats)`` for images + landmarks."""
        images = images.to(next(self.parameters()).device)
        landmarks = self._as_landmarks(
            landmarks, images.size(0), images.device, images.dtype
        )
        f_local = self.branches["local"](images)
        f_global = self.branches["global"](images)
        f_geom = self.branches["geometry"](landmarks)
        fused, weights = self.fusion(f_local, f_global, f_geom)
        return fused, weights, [f_local, f_global, f_geom]

    def forward(
        self, images: torch.Tensor, landmarks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused, weights, _ = self.encode(images, landmarks)
        return self.classifier(fused), weights


def build_tribranch_model(cfg) -> CompoundEmotionModel:
    """Factory mirroring the v1 ``build_model`` convention."""
    model = CompoundEmotionModel(
        num_classes=cfg.NUM_CLASSES,
        embed_dim=cfg.EMBED_DIM,
        dropout=cfg.DROPOUT,
        local_backbone=cfg.LOCAL_BACKBONE,
        vggface2_weights=cfg.VGGFACE2_WEIGHTS,
        hf_face_vit_id=cfg.HF_FACE_VIT_ID,
        num_landmarks=cfg.NUM_LANDMARKS,
        landmark_dim=cfg.LANDMARK_DIM,
        geometry_hidden=cfg.GEOMETRY_HIDDEN,
    )
    return model.to(cfg.DEVICE)
