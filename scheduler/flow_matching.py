"""Flow Matching 추론 핵심 함수들 — 추론 전용 (PIERROT_INFER).

추론 흐름:
    1) timestep_schedule.get_schedule()  schedule 생성
    2) (모델 forward 에서 velocity 또는 x̂_0 예측)
    3) x_prediction 모드면 x_prediction_to_velocity() 로 변환
    4) euler_step()                      한 step 진행 : x_{t-dt} = x_t - dt·v_t

prediction mode:
    "velocity"     : 모델이 v = ε - x_0 직접 예측 (표준 Flow Matching)
    "x_prediction" : 모델이 x_0 직접 예측 → 추론 시 v 로 역산.
"""
from typing import Literal

from torch import Tensor


# Prediction mode — Literal 로 단순화 (typo 방지 + IDE autocomplete)
PredictionMode = Literal["velocity", "x_prediction"]


def x_prediction_to_velocity(
    sample:   Tensor,
    x_0_pred: Tensor,
    t:        float,
    eps:      float = 0.05,
) -> Tensor:
    """추론 시 x̂_0 예측을 velocity 로 변환.   v = (x_t - x̂_0) / t.

    수식 유도:
        x_t = (1-t)·x_0 + t·ε   ⇒   ε = (x_t - (1-t)·x_0) / t
        v   = ε - x_0           ⇒   v = (x_t - x_0) / t

    eps clamp:
        t ≈ 0 에서 분모 발산을 막기 위해 t 를 max(t, eps) 로 보정.

    Args:
        sample   : (B, C, H, W)   현재 latent x_t
        x_0_pred : (B, C, H, W)   모델 출력 x̂_0
        t        : float          현재 timestep
        eps      : float          분모 보호 임계 (기본 0.05)

    Returns:
        (B, C, H, W)   환산된 velocity v̂
    """
    return (sample - x_0_pred) / max(t, eps)


def euler_step(sample: Tensor, velocity: Tensor, dt: float) -> Tensor:
    """Euler integration 한 step.   x_{t-dt} = x_t - dt · v_t.

    수식 의의:
        Flow Matching 의 ODE :  dx_t / dt = v_t
        Euler 1차 근사       :  x_{t-dt} ≈ x_t - dt · v_t  (음수 dt 로 t 감소 방향)

    Args:
        sample   : (B, C, H, W)   현재 latent x_t
        velocity : (B, C, H, W)   모델 예측 v_t (Flow Matching velocity)
        dt       : float          시간 간격 (양수, t_curr - t_prev)

    Returns:
        (B, C, H, W)   다음 latent x_{t-dt}
    """
    return sample - dt * velocity
