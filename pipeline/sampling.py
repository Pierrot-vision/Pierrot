"""PIERROT 추론(Sampling) 진입점 — Flow Matching 모델로부터 노이즈 → 이미지 생성.

흐름:
    1. timestep schedule 생성       (scheduler.timestep_schedule.get_schedule)
    2. 매 step 모델 forward         → velocity 또는 x̂_0 예측
    3. (x_prediction 모드면) x̂_0 → v̂ 변환
    4. Euler step 으로 x 업데이트   (scheduler.flow_matching.euler_step)

prediction_mode:
    "velocity"     모델이 v 직접 예측 (표준 Flow Matching, 기본)
    "x_prediction" 모델이 x_0 예측 → 추론 시 v 로 역산.   LPIPS / P-DINO 학습 시 사용

API:
    denoise()      — 일반 denoise 루프 (multi-reference 지원)
    denoise_cfg()  — Classifier-Free Guidance (cond + uncond 합성)
"""
import torch
from torch import Tensor

from ..scheduler.flow_matching import PredictionMode, euler_step, x_prediction_to_velocity
from ..scheduler.timestep_schedule import ScheduleMethod, get_schedule


#  내부 헬퍼 — schedule 생성 로직 공유

def _make_schedule(
    model,
    h: int,
    w: int,
    num_steps: int,
    method:    ScheduleMethod,
    shift:     float,
    max_t:     float = 1.0,
) -> list[float]:
    """이미지 H, W 와 method 에 맞춰 timestep schedule 생성.

    method='snr_shift' 면 image_seq_len 자동 계산 후 SNR shift.
    그 외 ('linear' / 'linear_shift') 면 linear + shift.

    max_t:
        학습 시 t 분포는 sigmoid(N(0,1)) — 중심 t=0.5, t=1.0 도달 거의 0%.
        추론 schedule 의 시작은 t=1.0 → 학습이 안 본 영역 → 첫 step 발산 → 누적 추상화.
        해결: schedule 만든 뒤 max_t (default 0.95) 로 linear rescale.
        시작 t=0.95 (학습 분포 안 ~95% 분위수) — 학습된 영역에서 denoise 진행.
        시작 latent (pure noise N(0,1)) 는 t=0.95 의 (1-t)·x_0 + t·ε 분포와 거의 동등 (t≈1).

    Args:
        model     : PIERROT 모델 (patch_size 속성 사용)
        h, w      : int          이미지 (latent) 높이·너비
        num_steps : int          디노이징 step 수
        method    : str          "linear" / "linear_shift" / "snr_shift"
        shift     : float        linear_shift 강도
        max_t     : float        시작 t cap (기본 1.0 = no rescale)

    Returns:
        list[float]   길이 num_steps + 1
    """
    if method == "snr_shift":
        patch         = getattr(model, "patch_size", 2)
        image_seq_len = (h // patch) * (w // patch)
        timesteps = get_schedule(num_steps=num_steps, image_seq_len=image_seq_len, method="snr_shift")
    else:
        timesteps = get_schedule(num_steps=num_steps, method=method, shift=shift)
    # 시작 t cap — 학습 분포 안으로 강제.   끝 t (=0.0) 는 그대로 유지하기 위해 linear rescale.
    if max_t < 1.0:
        timesteps = [max_t * t for t in timesteps]
    return timesteps


#  Denoise — Flow Matching 추론 루프 (multi-reference 지원)

def denoise(
    model,                                       # PIERROT 모델
    img: Tensor,                                 # (B, C, H, W)  초기 노이즈
    prompt_embeds: Tensor,                       # (B, L_txt, context_in_dim)
    *,
    num_steps:             int             = 50,
    schedule_method:       ScheduleMethod  = "snr_shift",
    schedule_shift:        float           = 1.0,
    prediction_mode:       PredictionMode  = "velocity",
    prompt_attention_mask: Tensor | None   = None,
    ref_latents:           list[Tensor] | None = None,
    ref_t_offsets:         list[int]    | None = None,
) -> Tensor:
    """Flow Matching denoise.   t=1 (노이즈) → t=0 (이미지) 방향으로 num_steps 번 Euler step.

    schedule_method 3가지:
        "linear"        단순 linear (shift=1.0 동등)
        "linear_shift"  linear + simple shift (schedule_shift 사용)
        "snr_shift"     해상도 적응형 SNR shift (기본·권장)

    prediction_mode:
        "velocity"      모델이 v 직접 출력 (표준)
        "x_prediction"  모델이 x̂_0 출력 → 매 step 자동으로 v 로 변환

    Multi-reference (선택):
        ref_latents 를 주면 모든 step 에서 ref 가 함께 attention 에 참여.
        None / [] 이면 표준 T2I 동작.

    Args:
        model                 : PIERROT 디노이저
        img                   : (B, C, H, W)   초기 노이즈
        prompt_embeds         : (B, L, D)      텍스트 embedding
        num_steps             : int            디노이징 step 수
        schedule_method       : str            schedule 방식
        schedule_shift        : float          linear_shift 용
        prediction_mode       : str            "velocity" | "x_prediction"
        prompt_attention_mask : (B, L) | None  text mask
        ref_latents           : list[(B, C, H_ref_i, W_ref_i)] | None   다중 참조 이미지 latent
        ref_t_offsets         : list[int] | None                        각 ref 의 4D RoPE t-축 오프셋

    Returns:
        (B, C, H, W)   디노이즈된 latent
    """
    bs, _, h, w   = img.shape
    device, dtype = img.device, img.dtype

    timesteps = _make_schedule(model, h, w, num_steps, schedule_method, schedule_shift)

    # zip(timesteps[:-1], timesteps[1:]): (t_curr, t_prev) 쌍.  시간 1.0 → 0.0 감소이므로 dt > 0
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # 학습/추론 정합 — 학습 timestep 이 float32 이므로 추론도 float32 t 로 sin/cos timestep_embedding
        # 정밀도 보장 (bf16 t 면 sin/cos 손실).
        t_vec = torch.full((bs,), t_curr, dtype=torch.float32, device=device)               # (B,) float32

        out = model(                                                                        # (B, C, H, W)  v 또는 x̂_0
            image_latent          = img,
            timestep              = t_vec,
            prompt_embeds         = prompt_embeds,
            prompt_attention_mask = prompt_attention_mask,
            ref_latents           = ref_latents,
            ref_t_offsets         = ref_t_offsets,
        )

        # x_prediction 이면 x̂_0 → v̂ 변환 (분기는 여기 한 곳만)
        velocity = x_prediction_to_velocity(img, out, t_curr) if prediction_mode == "x_prediction" else out

        img = euler_step(img, velocity, dt=t_curr - t_prev)                                 # x_{t-dt} = x_t - dt·v_t

    return img


#  Denoise CFG — Classifier-Free Guidance

def denoise_cfg(
    model,                                       # PIERROT 모델
    img: Tensor,                                 # (B, C, H, W)  초기 노이즈
    prompt_embeds_cond:   Tensor,                # (B, L_txt, D) 조건부 (실제 prompt)
    prompt_embeds_uncond: Tensor,                # (B, L_txt, D) 무조건부 (빈/null prompt)
    *,
    num_steps:                    int             = 50,
    guidance_scale:               float           = 4.0,
    schedule_method:              ScheduleMethod  = "snr_shift",
    schedule_shift:               float           = 1.0,
    prediction_mode:              PredictionMode  = "velocity",
    prompt_attention_mask_cond:   Tensor | None   = None,
    prompt_attention_mask_uncond: Tensor | None   = None,
    ref_latents:                  list[Tensor] | None = None,
    ref_t_offsets:                list[int]    | None = None,
) -> Tensor:
    """Classifier-Free Guidance denoise.   매 step model forward 1 회 (배치 복제로 cond+uncond 동시).

    수식:
        v̂ = v_uncond + guidance_scale · (v_cond - v_uncond)

    Args:
        model                        : PIERROT 디노이저
        img                          : (B, C, H, W)    초기 노이즈
        prompt_embeds_cond           : (B, L, D)       실제 prompt
        prompt_embeds_uncond         : (B, L, D)       null prompt
        num_steps                    : int             디노이징 step 수
        guidance_scale               : float           CFG 강도.   0=uncond, 1=cond, 보통 3~7
        schedule_method              : str             schedule 방식
        schedule_shift               : float           linear_shift 용
        prediction_mode              : str             "velocity" | "x_prediction"
        prompt_attention_mask_cond   : (B, L) | None   cond 마스크
        prompt_attention_mask_uncond : (B, L) | None   uncond 마스크
        ref_latents                  : list[(B, C, H_ref_i, W_ref_i)] | None
                                       다중 참조 이미지 latent (None / [] = 표준 T2I CFG).
                                       각 ref 는 cond/uncond 양쪽에 동일하게 페어링됨.
        ref_t_offsets                : list[int] | None
                                       각 ref 의 4D RoPE t-축 오프셋.

    Returns:
        (B, C, H, W)   디노이즈된 latent
    """
    bs, _, h, w   = img.shape
    device, dtype = img.device, img.dtype

    # 배치 복제 — 한 번의 forward 로 cond/uncond 동시 처리
    img_paired    = torch.cat([img, img], dim=0)                                          # (2B, C, H, W)
    prompt_paired = torch.cat([prompt_embeds_uncond, prompt_embeds_cond], dim=0)          # (2B, L, D)

    # 마스크 한쪽 누락 처리
    # 둘 다 None        → mask_paired=None
    # 한쪽만 있음       → 없는 쪽을 ones (전부 attended) 로 default 후 concat
    # 둘 다 있음        → 그대로 concat
    if prompt_attention_mask_cond is None and prompt_attention_mask_uncond is None:
        mask_paired = None
    else:
        L         = prompt_embeds_cond.shape[1]
        ones_mask = torch.ones(bs, L, dtype=torch.bool, device=device)
        cond_m    = prompt_attention_mask_cond   if prompt_attention_mask_cond   is not None else ones_mask
        uncond_m  = prompt_attention_mask_uncond if prompt_attention_mask_uncond is not None else ones_mask
        mask_paired = torch.cat([uncond_m, cond_m], dim=0)                                # (2B, L)

    # ref 페어링
    # ref_latents 의 각 (B, C, H_ref, W_ref) 텐서를 (2B, C, H_ref, W_ref) 로 페어링.
    # cond/uncond 가 같은 ref 를 보도록 — ref 자체는 prompt-independent.
    refs_paired = (
        [torch.cat([r, r], dim=0) for r in ref_latents]
        if ref_latents else None
    )

    timesteps = _make_schedule(model, h, w, num_steps, schedule_method, schedule_shift)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # 학습/추론 정합 — 학습이 float32 t 이므로 추론도 float32 강제 (bf16 t 면 sin/cos 손실).
        t_vec = torch.full((2 * bs,), t_curr, dtype=torch.float32, device=device)         # (2B,) float32

        out_paired = model(                                                               # (2B, C, H, W)  v 또는 x̂_0
            image_latent          = img_paired,
            timestep              = t_vec,
            prompt_embeds         = prompt_paired,
            prompt_attention_mask = mask_paired,
            ref_latents           = refs_paired,
            ref_t_offsets         = ref_t_offsets,
        )

        # x_prediction 이면 v 공간에서 CFG 합성하기 위해 먼저 변환
        if prediction_mode == "x_prediction":
            out_paired = x_prediction_to_velocity(img_paired, out_paired, t_curr)

        # uncond / cond 분리 후 CFG 합성 (둘 다 v 공간)
        v_uncond, v_cond = out_paired.chunk(2, dim=0)                                     # 각 (B, C, H, W)
        v_guided         = v_uncond + guidance_scale * (v_cond - v_uncond)                # (B, C, H, W)
        v_paired         = torch.cat([v_guided, v_guided], dim=0)                         # (2B, C, H, W)  페어 유지

        img_paired = euler_step(img_paired, v_paired, dt=t_curr - t_prev)

    # 두 배치는 같은 v_guided 가 적용되어 동일 — 첫 절반만 반환
    return img_paired.chunk(2, dim=0)[0]                                                  # (B, C, H, W)
