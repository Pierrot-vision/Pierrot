"""PIERROT_INFER: PIERROT 추론 전용 패키지.

PIERROT 코드베이스에서 추론에 필요한 코드만 추출한 standalone 패키지.
학습 코드 (train.py / dataset / optimizer / EMA / REPA / TREAD / LPIPS / args 등) 는 제외.

구성:
    model                     — PIERROT 트랜스포머 본체 (4D RoPE, multi-ref)
    scheduler.timestep_schedule — 추론 schedule (linear / linear_shift / snr_shift)
    scheduler.flow_matching     — euler_step / x_prediction_to_velocity (추론 함수만)
    pipeline.sampling           — 추론 코어 (denoise / denoise_cfg)
    pipeline.PIERROTPipeline    — 사용자 친화 wrapper
    text_encoding               — quote-aware 토크나이저 유틸
    constants                   — chi_prompt 등 추론 상수
    sample                      — CLI 진입점 (python -m PIERROT_INFER.sample)
"""

# 모델 (model/)
from .model import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    ModulationOut,
    PIERROT,
    PIERROTBlock,
    PIERROTDualBlock,
    PIERROTParams,
    QKNorm,
    RMSNorm,
    apply_rope,
    get_image_ids_4d,
    get_text_ids_4d,
    img2seq,
    seq2img,
    timestep_embedding,
)

# Scheduler (추론 전용)
from .scheduler import (
    PredictionMode,
    ScheduleMethod,
    compute_empirical_mu,
    euler_step,
    generalized_time_snr_shift,
    get_schedule,
    get_schedule_linear,
    get_schedule_snr_shift,
    shift_timesteps,
    x_prediction_to_velocity,
)

# 추론 코어 + wrapper (pipeline/)
from .pipeline import (
    DEFAULT_RESOLUTION,
    PIERROTPipeline,
    TextPreprocessor,
    classify_height_width_bin,
    denoise,
    denoise_cfg,
)

__all__ = [
    # model
    "PIERROT",
    "PIERROTParams",
    "img2seq",
    "seq2img",
    "PIERROTBlock",
    "PIERROTDualBlock",
    "Modulation",
    "ModulationOut",
    "LastLayer",
    "QKNorm",
    "RMSNorm",
    "MLPEmbedder",
    "EmbedND",
    "apply_rope",
    "timestep_embedding",
    "get_image_ids_4d",
    "get_text_ids_4d",
    # scheduler
    "PredictionMode",
    "ScheduleMethod",
    "get_schedule",
    "get_schedule_linear",
    "get_schedule_snr_shift",
    "shift_timesteps",
    "generalized_time_snr_shift",
    "compute_empirical_mu",
    "euler_step",
    "x_prediction_to_velocity",
    # pipeline
    "denoise",
    "denoise_cfg",
    "DEFAULT_RESOLUTION",
    "PIERROTPipeline",
    "TextPreprocessor",
    "classify_height_width_bin",
]
