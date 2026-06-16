"""PIERROT pipeline 진입점 — 추론 전용.

구성:
    PIERROT_INFER.pipeline.sampling          — 함수형 코어 (denoise / denoise_cfg)
    PIERROT_INFER.pipeline.PIERROTPipeline   — 사용자 친화 wrapper.   내부에서 sampling.denoise_cfg() 호출.
    PIERROT_INFER.sample             — CLI 진입점 (argparse + wiring, PIERROTPipeline 호출)
"""
# 함수형 코어
from .sampling import denoise, denoise_cfg

# 사용자 친화 wrapper
from .PIERROTPipeline import (
    DEFAULT_RESOLUTION,
    PIERROTPipeline,
    TextPreprocessor,
    classify_height_width_bin,
)

__all__ = [
    # 함수형 코어
    "denoise",
    "denoise_cfg",
    # wrapper
    "DEFAULT_RESOLUTION",
    "PIERROTPipeline",
    "TextPreprocessor",
    "classify_height_width_bin",
]
