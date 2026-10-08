"""Dual-domain tri-branch encoders (v2).

Branch 1 — :class:`LocalTextureEncoder` (local textures)
    ``efficientnet_b0`` or ``resnet50`` from :mod:`timm`, optionally warm-
    started from VGGFace2 weights, with the retained :class:`CBAMBlock`
    attention attached to the final-stage feature map. Projects to 512-D.

Branch 2 — :class:`GlobalContextEncoder` (global context)
    Vision Transformer pre-trained on facial expressions
    (``trpakov/vit-face-expression`` via Hugging Face ``transformers``,
    AffectNet-trained). The classification head is discarded; the ``[CLS]``
    token is projected to 512-D. Falls back to a timm ViT when the
    checkpoint cannot be downloaded (offline environments).

Branch 3 — :class:`GeometryMLP` (facial geometry)
    Replaces the removed ``ALSTM`` class entirely. A 3-layer MLP over the
    flattened 468 × 3 facial-landmark tensor (1404 → 1024 → 1024 → 512).

Every encoder returns a ``(B, embed_dim)`` vector and implements the
freeze/unfreeze helpers (``_freeze_all`` / ``unfreeze_last_n_blocks`` /
``unfreeze_all``) so legacy staged curricula keep working; the v2
two-phase curriculum itself uses :meth:`set_trainable`.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn

from src.attention import CBAMBlock

try:  # optional dependency — only needed to instantiate Branch 1 / fallbacks
    import timm
except ImportError:  # pragma: no cover
    timm = None

try:  # optional dependency — only needed to instantiate Branch 2
    from transformers import ViTModel
except ImportError:  # pragma: no cover
    ViTModel = None


# ─────────────────────────────────────────────────────────────────────────
# Branch 1 — Local textures (timm CNN + CBAM)
# ─────────────────────────────────────────────────────────────────────────
class LocalTextureEncoder(nn.Module):
    """Local-texture branch: timm CNN backbone + CBAM + 512-D projection.

    Args:
        backbone_name: timm model id, ``"efficientnet_b0"`` or ``"resnet50"``.
        vggface2_weights: optional VGGFace2 checkpoint used to warm-start the
            backbone — a local ``.pth``/``.bin`` path, an ``http(s)`` URL, or a
            Hugging Face ``"<repo>/<file>"`` id. Loaded with ``strict=False``
            so classifier-shape mismatches are tolerated. ``None`` keeps the
            standard timm (ImageNet) pre-trained weights.
        embed_dim: output embedding size (512).
        dropout: dropout rate in the projection head.
        pretrained: load timm pre-trained weights before any VGGFace2 load.
    """

    SUPPORTED = ("efficientnet_b0", "resnet50")

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        vggface2_weights: Optional[str] = None,
        embed_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for LocalTextureEncoder — `pip install timm`."
            )
        if backbone_name not in self.SUPPORTED:
            raise ValueError(
                f"backbone_name must be one of {self.SUPPORTED}, got {backbone_name!r}."
            )
        self.backbone_name = backbone_name
        # features_only → list of stage feature maps; CBAM goes on the last one
        # (i.e. the final residual/MBConv block output).
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        if vggface2_weights:
            self._load_vggface2_weights(vggface2_weights)

        last_ch = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAMBlock(last_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.LayerNorm(last_ch),
            nn.Dropout(dropout),
            nn.Linear(last_ch, embed_dim),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self._embed_dim = embed_dim
        self._freeze_all()

    # -- weight loading ---------------------------------------------------
    def _load_vggface2_weights(self, source: str) -> None:
        if os.path.isfile(source):
            state = torch.load(source, map_location="cpu")
        elif source.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(source, map_location="cpu")
        else:  # treat as "<hub-repo>/<file>" on Hugging Face
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Loading VGGFace2 weights from the HF Hub needs "
                    "`pip install huggingface_hub`."
                ) from exc
            if "/" not in source or source.count("/") < 2:
                raise ValueError(
                    "HF-hub VGGFace2 source must look like '<repo>/<file>', e.g. "
                    "'my-org/vggface2-resnet50/pytorch_model.bin'."
                )
            repo, fname = source.rsplit("/", 1)
            path = hf_hub_download(repo_id=repo, filename=fname)
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        print(
            f"  LocalTextureEncoder: VGGFace2 weights loaded from {source} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

    # -- freeze helpers (legacy staged-curriculum interface) --------------
    def _ordered_blocks(self) -> List[nn.Module]:
        blocks: List[nn.Module] = []
        if "resnet" in self.backbone_name:
            for layer in (
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ):
                blocks.extend(list(layer))
        else:  # efficientnet: blocks = stages of MBConv blocks + head
            for stage in self.backbone.blocks:
                blocks.extend(list(stage))
            blocks.append(self.backbone.conv_head)
        return blocks

    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 8):
        for block in self._ordered_blocks()[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"  LocalTextureEncoder unfrozen last {n} blocks — trainable: {tr:,}")

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  LocalTextureEncoder fully unfrozen")

    def set_trainable(self, trainable: bool):
        for p in self.parameters():
            p.requires_grad = trainable

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]          # (B, C, H, W) final-stage map
        feats = self.cbam(feats)              # retained spatial+channel attention
        feats = self.pool(feats).flatten(1)   # (B, C)
        return self.projection(feats)         # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────
# Branch 2 — Global context (facial-expression ViT)
# ─────────────────────────────────────────────────────────────────────────
class GlobalContextEncoder(nn.Module):
    """Global-context branch: AffectNet ViT, ``[CLS]`` → 512-D projection.

    Args:
        hf_model_id: Hugging Face checkpoint id of the facial-expression ViT
            (default ``"trpakov/vit-face-expression"``).
        embed_dim: output embedding size (512).
        dropout: dropout rate in the projection head.
        fallback_timm: timm ViT used when the HF checkpoint is unreachable.
    """

    def __init__(
        self,
        hf_model_id: str = "trpakov/vit-face-expression",
        embed_dim: int = 512,
        dropout: float = 0.3,
        fallback_timm: str = "vit_base_patch16_224",
    ):
        super().__init__()
        self.hf_model_id = hf_model_id
        self.backend: str = "hf"
        hidden_dim = 768
        try:
            if ViTModel is None:
                raise ImportError("transformers is not installed.")
            # ViTModel drops the classification head of the ...ForImageClassification
            # checkpoint and exposes last_hidden_state (incl. the [CLS] token).
            self.vit = ViTModel.from_pretrained(hf_model_id)
            hidden_dim = self.vit.config.hidden_size
            print(f"  GlobalContextEncoder: loaded HF checkpoint '{hf_model_id}'.")
        except Exception as exc:  # offline / missing dep → timm fallback
            if timm is None:
                raise ImportError(
                    "Neither the HF facial-expression ViT "
                    f"({exc}) nor the timm fallback is available. Install "
                    "`transformers` (online) or `timm`."
                ) from exc
            print(
                f"  GlobalContextEncoder: WARNING — could not load '{hf_model_id}' "
                f"({exc}); falling back to timm '{fallback_timm}'."
            )
            self.backend = "timm"
            self.vit = timm.create_model(fallback_timm, pretrained=True, num_classes=0)
            hidden_dim = self.vit.num_features

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
        )
        self._freeze_all()

    # -- freeze helpers ----------------------------------------------------
    def _vit_blocks(self):
        if self.backend == "hf":
            return list(self.vit.encoder.layer)
        return list(self.vit.blocks)

    def _freeze_all(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 8):
        for block in self._vit_blocks()[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for attr in ("layernorm", "norm", "pooler"):
            mod = getattr(self.vit, attr, None)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True
        tr = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        print(f"  GlobalContextEncoder unfrozen last {n} blocks — trainable: {tr:,}")

    def unfreeze_all(self):
        for p in self.vit.parameters():
            p.requires_grad = True
        print("  GlobalContextEncoder fully unfrozen")

    def set_trainable(self, trainable: bool):
        for p in self.parameters():
            p.requires_grad = trainable

    # -- forward -----------------------------------------------------------
    def _cls_token(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "hf":
            return self.vit(pixel_values=x).last_hidden_state[:, 0]
        return self.vit.forward_features(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self._cls_token(x))  # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────
# Branch 3 — Facial geometry MLP (replaces ALSTM)
# ─────────────────────────────────────────────────────────────────────────
class GeometryMLP(nn.Module):
    """Facial-geometry branch: 3-layer MLP over 468 × 3 landmarks.

    Accepts either a flattened ``(B, 1404)`` tensor or ``(B, 468, 3)`` and
    maps 1404 → 1024 → 1024 → 512 (three linear layers, 1024-D hidden).
    """

    def __init__(
        self,
        num_landmarks: int = 468,
        landmark_dim: int = 3,
        hidden_dim: int = 1024,
        embed_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_dim = num_landmarks * landmark_dim  # 1404
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def set_trainable(self, trainable: bool):
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        if landmarks.dim() == 3:                    # (B, 468, 3) → (B, 1404)
            landmarks = landmarks.flatten(start_dim=1)
        return self.net(landmarks.float())          # (B, embed_dim)
