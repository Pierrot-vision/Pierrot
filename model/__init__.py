"""PIERROT 모델 서브패키지 — 순수 PyTorch PIERROT 디퓨전 모델."""

from .pierrot import PIERROT, PIERROTParams, img2seq, seq2img
from .pierrot_modules import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    ModulationOut,
    PIERROTBlock,
    PIERROTDualBlock,
    QKNorm,
    RMSNorm,
    apply_rope,
    get_image_ids_4d,
    get_text_ids_4d,
    timestep_embedding,
)

__all__ = [
    # Top-level model
    "PIERROT",
    "PIERROTParams",
    # Utilities
    "img2seq",
    "seq2img",
    # Building blocks
    "PIERROTBlock",
    "PIERROTDualBlock",
    "Modulation",
    "ModulationOut",
    "LastLayer",
    "QKNorm",
    "RMSNorm",
    "MLPEmbedder",
    "EmbedND",
    # Functions (4D RoPE 전용)
    "apply_rope",
    "get_image_ids_4d",
    "get_text_ids_4d",
    "timestep_embedding",
]
