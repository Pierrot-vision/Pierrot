"""PIERROT 모델 config preset 모음 — 학습/추론 공용.

내용:
    _PIERROT_BASE   — preset 공통 default.
    CONFIG_PRESETS  — 'tiny' / '0.8b' / '1.6b' 학습용 모델 크기 preset.
                      크기 라벨은 24h 옵션(GQA n_kv=4 + adaln cap) 기준.   raw(MHA)는 더 큼.
"""
_PIERROT_BASE: dict = {
    "in_channels":            32,          # latent 채널 (FLUX.2 VAE 32ch)
    "patch_size":              2,          # 패치 크기 — patch_size=2 → (H/2, W/2) 토큰
    "context_in_dim":         7680,        # 텍스트 인코더 출력 차원 (Qwen3-4B + layer 9/18/27 fusion)
    "mlp_ratio":              3.5,         # MLP 확장 비율
    "theta":                  2000,        # RoPE 주파수 base (4D 모드)
    "time_factor":            1000.0,      # timestep 스케일 팩터
    "time_max_period":        10000,       # sinusoidal 주파수 스펙트럼 하한
    "conditioning_block_ids": None,
    "bottleneck_size":        None,
}

# 모든 preset 은 4D RoPE 전용 (axes_dim 길이 4 필수).
CONFIG_PRESETS: dict[str, dict] = {
    # 스모크 테스트용
    "tiny":  {**_PIERROT_BASE, "hidden_size":  256, "num_heads":  4, "depth":  2, "axes_dim": [16, 16, 16, 16]},

    # 표준 학습 (FLUX.2 VAE 기준) — 0.857B (raw MHA 1.46B)
    "0.8b": {
        **_PIERROT_BASE,
        "hidden_size":      1792,
        "num_heads":        28,
        "depth":            16,                # 전체 트랜스포머 블록 수
        "dual_block_count": 3,                 # 앞 N 개만 PIERROTDualBlock, 나머지는 PIERROTBlock
        "axes_dim":         [16, 16, 16, 16],  # 4축 (t, h, w, l) 채널 분배
    },

    # 깊이 성장 1단계 (0.8b 가중치 승계) — 너비 고정, depth 만 ↑.   1.618B
    "1.6b": {
        **_PIERROT_BASE,
        "hidden_size":      1792,              # 0.8b 와 동일 (가중치 shape 호환 필수)
        "num_heads":        28,                # 동일
        "depth":            33,                # 16 → 33 (≈ 1.618B)
        "dual_block_count": 3,                 # 동일
        "axes_dim":         [16, 16, 16, 16],
    },

}


__all__ = ["CONFIG_PRESETS", "_PIERROT_BASE"]
