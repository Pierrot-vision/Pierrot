"""추론 시 timestep schedule 생성 — 함수형 버전.

Flow Matching 추론은 t=1.0 (순수 노이즈) → t=0.0 (이미지) 방향으로 N step 진행한다.
N+1 개의 timestep 을 [0, 1] 시간 축 어디에 배치할지가 'schedule' 의 본질.

세 가지 방식 제공:
    1) "linear"        단순 linear
    2) "linear_shift"  linear + simple shift (shift 단일 파라미터로 단순 제어)
    3) "snr_shift"     해상도 적응형 SNR shift (이미지 토큰 수 기반 mu 자동)

통합 인터페이스 `get_schedule(method=...)` 로 선택 사용.
"""
import math
from typing import Literal

import torch
from torch import Tensor


#  공통 유틸 — simple time shift

def shift_timesteps(timesteps: Tensor, shift: float) -> Tensor:
    """Simple time shift.   t' = shift · t / (1 + (shift - 1) · t).

    효과:
        shift = 1.0 :  변화 없음 (linear 그대로)
        shift > 1.0 :  t 가 t=1 쪽으로 밀림 (큰 t 가 더 많아짐)
        shift < 1.0 :  t 가 t=0 쪽으로 밀림 (작은 t 가 더 많아짐)

    Args:
        timesteps : (N+1,)   원본 timestep (보통 linspace(1, 0, N+1))
        shift     : float    shift 강도.   1.0 이면 no-op

    Returns:
        (N+1,)   shift 적용된 timestep
    """
    if shift == 1.0:
        return timesteps
    return shift * timesteps / (1 + (shift - 1) * timesteps)


#  Linear + simple shift

def get_schedule_linear(num_steps: int, shift: float = 1.0) -> list[float]:
    """Linear timestep schedule.   linear(1→0) 후 shift_timesteps 적용.

    Args:
        num_steps : int    디노이징 step 수
        shift     : float  shift 강도 (기본 1.0 = no-shift = pure linear)

    Returns:
        list[float]   길이 num_steps + 1.   [1.0, ..., 0.0]
    """
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)               # (N+1,)
    timesteps = shift_timesteps(timesteps, shift)
    return timesteps.tolist()


#  해상도 적응형 SNR shift

def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float = 1.0) -> Tensor:
    """SNR 기반 timestep 재분배.   t' = e^μ / (e^μ + ((1-t)/t)^σ).

    수식 의의:
        SNR(t) = (1-t)/t   (Flow Matching 의 신호/노이즈 비)
        t' = e^μ / (e^μ + SNR(t)^σ)
        → mu 가 클수록 t=1 (낮은 SNR, 노이즈 많음) 영역에 더 많은 step 압축

    Args:
        t     : (N+1,)   원본 timestep ∈ [0, 1]
        mu    : float    shift 강도.   compute_empirical_mu 로 자동 도출 가능
        sigma : float    shift 모양.   1.0 이면 부드러운 시그모이드 (기본)

    Returns:
        (N+1,)   SNR-shifted timestep
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """이미지 토큰 수 + step 수로부터 mu 를 자동 도출.

    1024px 학습·추론 실험에서 찾은 경험적 상수 사용.
    큰 이미지일수록 mu 가 커져서 t=1 근처에 step 이 더 몰림 (어려운 영역 집중).

    Args:
        image_seq_len : int   이미지 토큰 수 = (H/patch) * (W/patch).   1024px+patch=2 → 4096
        num_steps     : int   디노이징 step 수

    Returns:
        float   `generalized_time_snr_shift` 의 mu 인자로 사용
    """
    # 경험적 상수
    a1, b1 = 8.73809524e-05, 1.89833333    # 작은 이미지·작은 step 영역
    a2, b2 = 0.00016927,    0.45666666     # 큰 이미지·큰 step 영역

    # 큰 이미지 (≈ 1024px+) : 단순 선형 모델
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    # 작은 이미지: image_seq_len 고정 시 num_steps 에 따라 보간
    m_200 = a2 * image_seq_len + b2        # num_steps=200 일 때 추정 mu
    m_10  = a1 * image_seq_len + b1        # num_steps=10  일 때 추정 mu
    a = (m_200 - m_10) / 190.0             # (200 - 10 = 190)
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule_snr_shift(num_steps: int, image_seq_len: int) -> list[float]:
    """SNR-shift timestep schedule.   linear → SNR shift.

    Args:
        num_steps     : int   디노이징 step 수 (예: 50)
        image_seq_len : int   이미지 토큰 수 = (H/patch) * (W/patch)

    Returns:
        list[float]   길이 num_steps + 1.   t=1 근처에 더 압축된 분포
    """
    mu        = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, sigma=1.0)
    return timesteps.tolist()


#  통합 인터페이스

ScheduleMethod = Literal["linear", "linear_shift", "snr_shift"]


def get_schedule(
    num_steps:     int,
    image_seq_len: int | None = None,
    method:        ScheduleMethod = "snr_shift",
    shift:         float = 1.0,
) -> list[float]:
    """Timestep schedule 생성 통합 인터페이스.   세 가지 방식 중 선택.

    Args:
        num_steps     : int           디노이징 step 수 (예: 50)
        image_seq_len : int | None    이미지 토큰 수.   method="snr_shift" 시 필수
        method        : str           "linear" / "linear_shift" / "snr_shift"
        shift         : float         method="linear_shift" 일 때 shift 강도 (기본 1.0 = no-op)

    Returns:
        list[float]   길이 num_steps + 1.   ts[0]=1.0, ts[-1]=0.0

    예시:
        # 1024px (patch=2 → 4096 토큰), SNR-shift 방식
        ts = get_schedule(num_steps=50, image_seq_len=4096, method="snr_shift")

        # linear + shift=3.0
        ts = get_schedule(num_steps=50, method="linear_shift", shift=3.0)

        # 베이스라인 (단순 linear)
        ts = get_schedule(num_steps=50, method="linear")
    """
    if method == "linear":
        return get_schedule_linear(num_steps, shift=1.0)

    if method == "linear_shift":
        return get_schedule_linear(num_steps, shift=shift)

    if method == "snr_shift":
        if image_seq_len is None:
            raise ValueError(
                "method='snr_shift' 는 image_seq_len 이 필요하다.  "
                "이미지 토큰 수 = (H/patch_size) * (W/patch_size) 로 계산해서 전달."
            )
        return get_schedule_snr_shift(num_steps, image_seq_len)

    raise ValueError(
        f"Unknown method: {method!r}.  "
        f"Use one of: 'linear', 'linear_shift', 'snr_shift'"
    )
