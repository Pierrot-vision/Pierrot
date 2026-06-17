"""PIERROT model layers 

구성:
    - 4D RoPE (t, h, w, l)  + EmbedND + apply_rope
    - 소형 모듈            : RMSNorm / QKNorm / MLPEmbedder
    - AdaLN-Zero           : Modulation / ModulationOut
    - 비대칭 attention 블록: PIERROTBlock  (image-only Q, text/ref KV-only)
    - 양방향 블록          : PIERROTDualBlock (text 도 Q 생성·정제)
    - 최종 projection      : LastLayer
"""
import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_image_ids_4d(
    bs: int, h: int, w: int, patch_size: int, device: torch.device, t_offset: int = 0,
) -> Tensor:
    """이미지 토큰의 4D 좌표 (t, h, w, l).  메인=t_offset 0, ref=10/20/30..., l=0 (dummy).

    출력 shape: (B, N, 4)  where N = (H/p) * (W/p)
    """
    H, W = h // patch_size, w // patch_size

    img_ids         = torch.zeros(H, W, 4, device=device)            # (H, W, 4)
    img_ids[..., 0] = t_offset                                       # t  — 이미지/참조 구분 축
    img_ids[..., 1] = torch.arange(H, device=device)[:, None]        # h  — row 좌표 (broadcast over W)
    img_ids[..., 2] = torch.arange(W, device=device)[None, :]        # w  — col 좌표 (broadcast over H)
    # img_ids[..., 3] = 0  ← 이미 0 (l 축 dummy)

    return img_ids.reshape(H * W, 4).unsqueeze(0).repeat(bs, 1, 1)   # (H*W, 4) → (1, N, 4) → (B, N, 4)


def get_text_ids_4d(bs: int, l_txt: int, device: torch.device, t_offset: int = 0) -> Tensor:
    """텍스트 토큰의 4D 좌표 (t, h=0, w=0, l=token_pos).

    출력 shape: (B, L_txt, 4)
    """
    txt_ids         = torch.zeros(l_txt, 4, device=device)           # (L_txt, 4)
    txt_ids[..., 0] = t_offset                                       # t  — 이미지와 묶을 때 일치
    # txt_ids[..., 1] = 0  ← 이미 0 (h dummy)
    # txt_ids[..., 2] = 0  ← 이미 0 (w dummy)
    txt_ids[..., 3] = torch.arange(l_txt, device=device)             # l  — 어순 위치

    return txt_ids.unsqueeze(0).repeat(bs, 1, 1)                     # (L_txt, 4) → (1, L_txt, 4) → (B, L_txt, 4)


def apply_rope(xq: Tensor, freqs_cis: Tensor) -> Tensor:
    """RoPE 회전 행렬 적용.

    입력  xq        : (B, H, L, D)              D = head_dim
          freqs_cis : (B, 1, L, D/2, 2, 2)      토큰·채널별 2x2 회전 행렬
    출력            : (B, H, L, D)
    """
    # .float() — RoPE 회전은 항상 fp32 (bf16/fp16 누적 시 drift).
    # .reshape(..., -1, 1, 2) — D → (D/2, 1, 2).   2 = 회전 평면의 (real, imag) 쌍.
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)                                       # (B, H, L, D/2, 1, 2)
    # freqs_cis[..., 0]: (B, 1, L, D/2, 2)  ← 회전행렬의 첫 번째 행 [cos, -sin]
    # freqs_cis[..., 1]: (B, 1, L, D/2, 2)  ← 두 번째 행                [sin,  cos]
    # broadcast 곱셈 — head 축 1 → num_heads 로 자동 확장.   2x2 회전행렬을 (real, imag) 페어에 적용.
    out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]                  # (B, H, L, D/2, 2)
    # reshape(*xq.shape) — (D/2, 2) 를 다시 D 로 평탄화.   .type_as(xq) — 원본 dtype 복원.
    return out.reshape(*xq.shape).type_as(xq)                                                # (B, H, L, D)


def compute_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    """Attention 계산 — SDPA wrapper.  CUDA + (fp16|bf16) 일 때 CuDNN backend 우선, 실패 시 fallback.

    입력  q, k, v  : (B, H, L_q 또는 L_kv, D)
          attn_mask: None 또는 (B, H, L_q, L_kv) 의 bool visibility mask/additive bias
    출력           : (B, H, L_q, D)
    """
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            # sdpa_kernel(CUDNN_ATTENTION) — H100/A100 에서 가장 빠른 backend.
            #   bf16/fp16 + contiguous + CUDA 조건 만족해야 함.   안 맞으면 RuntimeError.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                # F.scaled_dot_product_attention — torch 2.x SDPA.   내부적으로 backend 자동 선택,
                #   여기선 sdpa_kernel context 로 cuDNN 강제.   .contiguous() 는 memory layout 정합 (cuDNN 요구).
                return F.scaled_dot_product_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attn_mask,
                )
        except RuntimeError:
            pass
    # 폴백 — context 없이 PyTorch default backend (flash_attention / math 자동 선택).
    return F.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attn_mask,
    )


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """Flow Matching 의 연속 timestep → sinusoidal embedding.

    입력  t           : (B,)        연속값 timestep (예: 0~1)
          dim         : int         출력 임베딩 차원
          max_period  : int         주파수 스펙트럼 하한 제어
          time_factor : float       t 의 동적 범위 확장 (기본 1000 → 0~1000 범위)
    출력              : (B, dim)
    """
    t    = time_factor * t                                                                            # (B,)
    half = dim // 2
    
    # NOTE: float32 사용 (Apple MPS / 일부 NPU 가 float64 미지원).
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)  # (dim/2,)

    args = t[:, None].float() * freqs[None]                                                           # (B, 1) × (1, dim/2) → (B, dim/2)
    emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)                                      # (B, dim)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)                                  # 홀수 차원이면 0 추가
    return emb


class EmbedND(nn.Module):
    """N축 분리 RoPE.  sum(axes_dim) == head_dim 필수.  PIERROT 는 axes_dim=[t,h,w,l] 4축.

    옵션:
        axes_max_len : freqs precompute cache 의 각 축 최대 길이.
                       None — cache 비활성, 매 forward cos/sin 재계산.
                       list[int] — 첫 forward 에서 (max_len, dim/2, 2, 2) 행렬을 한 번 만들고
                                   이후 forward 는 정수 인덱싱(gather) 만으로 RoPE 생성.
                       길이 제약: len(axes_max_len) == len(axes_dim).

    cache 는 일반 list 속성 (nn.Buffer / nn.Parameter X) → state_dict 에 저장되지 않음.
    """

    def __init__(
        self,
        dim: int,
        theta: int,
        axes_dim: list[int],
        axes_max_len: list[int] | None = None,
    ):
        super().__init__()
        self.dim          = dim                   # head_dim 과 동일 (= sum(axes_dim))
        self.theta        = theta                 # RoPE 주파수 base
        self.axes_dim     = axes_dim              # 각 축 채널 수.  예: [16, 16, 16, 16]
        self.axes_max_len = axes_max_len          # 각 축의 max 좌표 (None=cache 미사용)

        # cache lazy init — 첫 forward 에서 device 결정 후 _build_cache 가 채움.
        self._freqs_cache: list[Tensor] | None = None

    def rope(self, pos: Tensor, dim: int, theta: int) -> Tensor:
        """1D RoPE 한 축 (cache 미사용 경로).

        입력   pos   : (B, N)                  이 축의 정수 좌표
              dim   : int                     이 축에 할당된 채널 수 (짝수)
              theta : int                     주파수 base
        출력         : (B, N, dim/2, 2, 2)      토큰·채널별 2x2 회전 행렬
        """
        assert dim % 2 == 0

        # NOTE: float32 사용 (Apple MPS / 일부 NPU 가 float64 미지원).
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim  # (dim/2,)
        omega = 1.0 / (theta ** scale)                                                 # (dim/2,)
        out   = pos.unsqueeze(-1) * omega.unsqueeze(0)                                 # (B, N, dim/2)

        # 2x2 회전행렬 R(θ) = [[cos, -sin], [sin, cos]] 의 4원소 stack
        out   = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()

    def _build_cache(self, device: torch.device) -> None:
        """axes_max_len 기반 freqs cache 빌드.   각 축 (max_len, d_i/2, 2, 2) 행렬을 한 번만 계산."""
        assert self.axes_max_len is not None

        cache: list[Tensor] = []
        for d, max_len in zip(self.axes_dim, self.axes_max_len):
            assert d % 2 == 0
            scale = torch.arange(0, d, 2, dtype=torch.float32, device=device) / d
            omega = 1.0 / (self.theta ** scale)
            pos   = torch.arange(max_len, dtype=torch.float32, device=device)
            ang   = pos[:, None] * omega[None, :]
            mat4  = torch.stack(
                [torch.cos(ang), -torch.sin(ang), torch.sin(ang), torch.cos(ang)], dim=-1,
            )
            mat   = rearrange(mat4, "p d (i j) -> p d i j", i=2, j=2).float()
            cache.append(mat)
        self._freqs_cache = cache

    def forward(self, ids: Tensor) -> Tensor:
        """ids:(B, N, n_axes) → pe:(B, 1, N, head_dim/2, 2, 2)  ← head 축 broadcast 용.

        분기:
            axes_max_len=None — rope() 매 호출 (cos/sin 재계산).
            axes_max_len=list — 첫 호출에서 cache 빌드, 이후 정수 인덱싱(gather) 만.
        """
        # Cache 미사용 경로 — rope() 매 호출.
        if self.axes_max_len is None:
            emb = torch.cat(
                [self.rope(ids[:, :, i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
                dim=-3,
            )
            return emb.unsqueeze(1)

        # Cache 사용 경로 — 첫 호출 또는 device 변경 시 재구축.
        if self._freqs_cache is None or self._freqs_cache[0].device != ids.device:
            self._build_cache(ids.device)

        # 안전 검증 — ids 좌표가 max_len 을 넘으면 명시적 에러.
        for i, max_len in enumerate(self.axes_max_len):
            ids_i_max = int(ids[:, :, i].max().item())
            if ids_i_max >= max_len:
                raise ValueError(
                    f"EmbedND cache: 축 {i} 의 ids 최대값 {ids_i_max} 이 "
                    f"axes_max_len[{i}]={max_len} 을 초과.   axes_max_len 을 늘리거나 "
                    f"ids 를 줄이세요 (ref t_offset 이 너무 클 가능성)."
                )

        # 정수 인덱싱(gather) — _freqs_cache[i]: (max_len, d_i/2, 2, 2)
        parts = [
            self._freqs_cache[i][ids[:, :, i].long()]                                                     # (B, N, d_i/2, 2, 2)
            for i in range(ids.shape[-1])
        ]
        emb = torch.cat(parts, dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    """SiLU 2-layer MLP — sinusoidal timestep(256) → hidden 사영용."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer  = nn.Linear(in_dim, hidden_dim,     bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """입력: (B, in_dim)   출력: (B, hidden_dim)."""
        return self.out_layer(F.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    """RMSNorm — γ (scale) 만 학습 (β 없음).

    옵션:
        affine : True (기본) — `scale` 학습 파라미터 1개.
                 False      — 학습 파라미터 0 (LayerNorm(elementwise_affine=False) 의 RMS 버전).

    수식:
        affine=True  : y = x · rsqrt(mean(x²) + ε) · scale
        affine=False : y = x · rsqrt(mean(x²) + ε)
    """

    def __init__(self, dim: int, affine: bool = True):
        super().__init__()
        # affine=False 일 때 scale 자체를 만들지 않아 state_dict 에 안 들어감.
        self.affine = affine
        if affine:
            self.scale = nn.Parameter(torch.ones(dim))            # 채널별 학습가능 scale (γ)

    def forward(self, x: Tensor) -> Tensor:
        """입력: (..., dim)   출력: (..., dim)  — shape 유지."""
        x_dtype = x.dtype
        rrms    = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        out     = x.float() * rrms
        if self.affine:
            out = out * self.scale                                 # γ 곱 (affine=True 일 때만)
        return out.to(x_dtype)


def _make_block_norm(dim: int, use_rmsnorm: bool) -> nn.Module:
    """블록의 pre/post norm 생성.   use_rmsnorm=True → affine 없는 RMSNorm, False → LayerNorm."""
    if use_rmsnorm:
        return RMSNorm(dim, affine=False)
    return nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)


class QKNorm(nn.Module):
    """Q, K 각자 RMSNorm — softmax 직전 norm 폭주 방지 (bf16 안정성)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm   = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """입력: q, k, v  각 (B, H, L, D).   출력: 정규화된 q, k  (B, H, L, D).  v 는 dtype 정합용."""
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


@dataclass
class ModulationOut:
    """AdaLN-Zero 한 경로분 (shift, scale, gate).  각 (B, 1, d).

    shift 타입 주의:
        기본은 Tensor 지만 adaln_4param=True 일 때는 scalar `0.0` (Python float) 가 들어온다.
        호출부 `(1+scale) · LN(x) + shift` 가 Tensor + float broadcasting 을 자연스럽게 처리한다.
    """
    shift: Tensor | float   # (B, 1, d) Tensor — adaln_4param=True 면 0.0 (scalar broadcast)
    scale: Tensor           # (B, 1, d)
    gate:  Tensor           # (B, 1, d)


class Modulation(nn.Module):
    """Zero 초기화 AdaLN.  vec(B, d) → (shift, scale, gate) × 2 (attn 경로 + mlp 경로).

    옵션:
        use_tanh_gate : gate 출력 tanh 클램프 (±1).
                        False (기본) — raw gate.
                        True       — gate.tanh().  tanh(0)=0 이라 zero-init 의 identity 유지.

        adaln_4param  : 4-param AdaLN — shift 항 제거.
                        False (기본) — (shift, scale, gate) × 2 = 6 param.   Linear `d → 6d`.
                        True       — (scale, gate) × 2 = 4 param.   Linear `d → 4d`.
                                     shift 자리는 scalar 0.0 으로 채워 호출부 변경 없이 동작.

        vec_dim       : modulation Linear 입력 차원.
                        None (기본) — vec_dim = dim.
                        int        — Linear `vec_dim → n_chunks·dim`.
    """

    def __init__(
        self,
        dim: int,
        use_tanh_gate: bool = False,
        adaln_4param: bool = False,
        vec_dim: int | None = None,
    ):
        super().__init__()

        self.use_tanh_gate = use_tanh_gate
        self.adaln_4param  = adaln_4param

        # 한 경로당 chunk 수: 3 (shift, scale, gate) 또는 2 (scale, gate).   양쪽 path (attn + mlp) 합산.
        self._n_per_path = 2 if adaln_4param else 3
        self._n_chunks   = self._n_per_path * 2                             # 4 or 6

        # Linear 입력 차원 = vec_dim (None=dim).
        in_dim           = vec_dim if vec_dim is not None else dim
        self.lin         = nn.Linear(in_dim, self._n_chunks * dim, bias=True)  # vec_dim → n_chunks·d

        # Zero init: 학습 시작 시 모든 modulation 0 → 블록이 identity 로 동작.
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut]:
        """입력: vec (B, d)   출력: (mod_attn, mod_mlp).  각 ModulationOut 의 필드 (B, 1, d)."""
        # (B, n_chunks·d) → (B, 1, n_chunks·d) → n_chunks × (B, 1, d).   n_chunks=4 or 6
        out = self.lin(F.silu(vec))[:, None, :].chunk(self._n_chunks, dim=-1)

        # adaln_4param 분기 — shift 분리 (4-param 시 scalar 0.0).
        if self.adaln_4param:
            scale_a, gate_a, scale_m, gate_m = out
            shift_a: float = 0.0   # type: ignore[assignment]
            shift_m: float = 0.0   # type: ignore[assignment]
        else:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = out

        # gate tanh 클램프 (use_tanh_gate=True 일 때만).
        if self.use_tanh_gate:
            return (
                ModulationOut(shift=shift_a, scale=scale_a, gate=gate_a.tanh()),    # attn 경로
                ModulationOut(shift=shift_m, scale=scale_m, gate=gate_m.tanh()),    # mlp 경로
            )

        return (
            ModulationOut(shift=shift_a, scale=scale_a, gate=gate_a),    # attn 경로
            ModulationOut(shift=shift_m, scale=scale_m, gate=gate_m),    # mlp 경로
        )


class PIERROTBlock(nn.Module):
    """비대칭 attention 트랜스포머 블록.

    구조:
        AdaLN → MMA(img-Q ↔ [txt; img]-KV) → +residual
        AdaLN → GEGLU MLP                  → +residual

    핵심:
        - img 만 Q 생성, txt 는 KV 만 (1B 경량성)
        - QKNorm (Q, K 각각 RMSNorm)
        - 4D RoPE 는 img Q/K 에만 적용 (txt 어순은 인코더 출력 신뢰)

    Forward 인자:
        img : (B, L_img, d)
        txt : (B, L_txt, d)
        vec : (B, d)                      timestep embedding
        pe  : (B, 1, L_img, D/2, 2, 2)    이미지 토큰용 RoPE
        attention_mask : None 또는 (B, L_txt) — 텍스트 패딩 0/1 마스크
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        sandwich_norm: bool = False,
        use_tanh_gate: bool = False,
        adaln_4param: bool = False,
        use_rmsnorm: bool = False,
        adaln_input_dim: int | None = None,
        n_kv_heads: int | None = None,                 # GQA — K/V head 수.  None=num_heads (MHA)
    ):
        super().__init__()
        self.hidden_size     = hidden_size
        self.num_heads       = num_heads

        # K/V head 수 — None 이면 num_heads (MHA), int 면 GQA.
        # group_size = num_heads / n_kv_heads (Q heads 가 K/V head 1개 공유하는 비율).
        self.n_kv_heads      = n_kv_heads if n_kv_heads is not None else num_heads
        self.group_size      = num_heads // self.n_kv_heads
        self.head_dim        = hidden_size // num_heads
        self.scale           = qk_scale or self.head_dim ** -0.5
        self.mlp_hidden_dim  = int(hidden_size * mlp_ratio)
        self.sandwich_norm   = sandwich_norm
        self.use_tanh_gate   = use_tanh_gate
        self.adaln_4param    = adaln_4param
        self.use_rmsnorm     = use_rmsnorm
        self.adaln_input_dim = adaln_input_dim

        # 이미지 Q/K/V
        # GQA: Q 는 num_heads 풀 사이즈, K/V 는 n_kv_heads 작은 사이즈.
        # qkv_out_dim = head_dim · (num_heads + 2 · n_kv_heads).   n_kv_heads=num_heads 면 기존 3d.
        qkv_out_dim       = self.head_dim * (num_heads + 2 * self.n_kv_heads)
        self.img_pre_norm = _make_block_norm(hidden_size, use_rmsnorm)
        self.img_qkv_proj = nn.Linear(hidden_size, qkv_out_dim, bias=False)   # (B, L_img, d) → (B, L_img, head_dim·(n+2·n_kv))
        self.attn_out     = nn.Linear(hidden_size, hidden_size, bias=False)   # (B, L_img, d) → (B, L_img, d)
        self.qk_norm      = QKNorm(self.head_dim)                             # Q, K 각자 RMSNorm

        # 텍스트 K/V (Q 없음 — 비대칭 attention 핵심)
        kv_out_dim       = self.head_dim * 2 * self.n_kv_heads
        self.txt_kv_proj = nn.Linear(hidden_size, kv_out_dim, bias=False)     # (B, L_txt, d) → (B, L_txt, 2·n_kv·head_dim)
        self.k_norm      = RMSNorm(self.head_dim)                             # 텍스트 K 만 RMSNorm

        # Sandwich-Norm
        # False (기본): Identity → Pre-Norm 만 사용.
        # True       : 각 잔차 합산 직전에 RMSNorm 한 번 더 → 진폭 누적 차단.
        self.attn_post_norm = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()
        self.ffn_post_norm  = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()

        # 참조(reference) 이미지 K/V — 메인과 별도 가중치, Q 없음
        # 메인과 ref 가중치 분리 → "ref 는 condition 만 제공" 을 weight 차원에서 강제.
        # RoPE 는 적용 — 각 ref 가 다른 t_offset 으로 회전돼 attention 이 자동 분리.
        self.ref_pre_norm = _make_block_norm(hidden_size, use_rmsnorm)
        self.ref_kv_proj  = nn.Linear(hidden_size, kv_out_dim, bias=False)                  # (B, L_ref, d) → (B, L_ref, 2·n_kv·head_dim)
        self.ref_k_norm   = RMSNorm(self.head_dim)                                          # ref K 만 RMSNorm

        # MLP — GEGLU (gate_proj·GELU ⊙ up_proj → down_proj)
        self.post_attention_layernorm = _make_block_norm(hidden_size, use_rmsnorm)
        self.gate_proj = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.mlp_act   = nn.GELU(approximate="tanh")

        # AdaLN-Zero conditioner.
        self.modulation = Modulation(
            hidden_size,
            use_tanh_gate=use_tanh_gate,
            adaln_4param=adaln_4param,
            vec_dim=adaln_input_dim,
        )

    def _compute_img_qkv(
        self,
        img: Tensor,
        pe:  Tensor,
        mod: ModulationOut,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """메인 이미지 Q/K/V 생성 + RoPE (AdaLN → qkv_proj → split → QKNorm → RoPE).

        GQA: Q 는 num_heads, K/V 는 n_kv_heads.

        Args:
            img: (B, L_img, d) 입력 이미지 토큰.
            pe:  (B, 1, L_img, D/2, 2, 2) 4D RoPE 회전행렬.
            mod: AdaLN-Zero (1+scale)·LN(x)+shift 의 scale/shift, 각 (B, 1, d).

        Returns:
            img_q: (B, num_heads,  L_img, D) — RoPE 적용 + QKNorm
            img_k: (B, n_kv_heads, L_img, D) — RoPE 적용 + QKNorm
            img_v: (B, n_kv_heads, L_img, D)
        """
        # self.img_pre_norm — LayerNorm 또는 RMSNorm (affine=False, AdaLN-Zero (1+scale)·LN(x)+shift 패턴).
        img_mod  = (1 + mod.scale) * self.img_pre_norm(img) + mod.shift                       # (B, L_img, d)
        # self.img_qkv_proj — nn.Linear(hidden → head_dim·(n+2·n_kv)).   bias=False (Z-Image 표준).
        img_qkv  = self.img_qkv_proj(img_mod)                                                 # (B, L_img, head_dim·(n+2·n_kv))
        q_dim    = self.head_dim * self.num_heads
        kv_dim_1 = self.head_dim * self.n_kv_heads

        # tensor.split([q_dim, kv_dim_1, kv_dim_1], dim=-1) — Q/K/V 비율로 3 분할 (GQA 차원).
        img_q_flat, img_k_flat, img_v_flat = img_qkv.split([q_dim, kv_dim_1, kv_dim_1], dim=-1)

        # einops.rearrange "B L (H D) -> B H L D" — head 축 분리.   H 가 다름 (Q vs K/V).
        img_q = rearrange(img_q_flat, "B L (H D) -> B H L D", H=self.num_heads)               # (B, n_heads, L_img, D)
        img_k = rearrange(img_k_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)              # (B, n_kv,    L_img, D)
        img_v = rearrange(img_v_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)              # (B, n_kv,    L_img, D)

        # self.qk_norm(QKNorm) — Q/K 각자 RMSNorm.   v 인자는 dtype 정합용 (출력 dtype 을 v 기준으로 cast).
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)                                      # head_dim 단위 RMSNorm
        # apply_rope() — 위 정의.   pe 의 head 축 1 → num_heads/n_kv_heads 로 자동 broadcast.
        img_q        = apply_rope(img_q, pe)                                                  # (B, n_heads, L_img, D)
        img_k        = apply_rope(img_k, pe)                                                  # (B, n_kv,    L_img, D)
        return img_q, img_k, img_v

    def _compute_txt_kv(self, txt: Tensor) -> tuple[Tensor, Tensor]:
        """텍스트 K/V 생성 (Q 없음, 비대칭 attention 의 핵심).   RoPE 미적용 (인코더 PE 신뢰).

        Args:
            txt: (B, L_txt, d) 텍스트 인코더 출력.

        Returns:
            txt_k: (B, n_kv_heads, L_txt, D) — RMSNorm 적용
            txt_v: (B, n_kv_heads, L_txt, D)
        """
        # self.txt_kv_proj — nn.Linear(hidden → 2·n_kv·head_dim).   K/V 만.
        txt_kv       = self.txt_kv_proj(txt)                                                  # (B, L_txt, 2·n_kv·head_dim)
        # rearrange "B L (K H D) -> K B H L D" — K=2 (K/V) 로 unstack 후 unpacking.
        txt_k, txt_v = rearrange(txt_kv, "B L (K H D) -> K B H L D", K=2, H=self.n_kv_heads)  # 2 × (B, n_kv, L_txt, D)
        # self.k_norm — RMSNorm(head_dim).   텍스트 K 만 정규화 (V 는 정규화 안 함).
        txt_k        = self.k_norm(txt_k)                                                     # head_dim 단위 RMSNorm
        return txt_k, txt_v

    def _compute_ref_kv(
        self,
        refs:    Tensor | None,
        pe_refs: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """참조 이미지 K/V 생성 + RoPE.   refs 가 None 이거나 빈 경우 (None, None) 반환.

        각 ref 의 t_offset 으로 RoPE 회전 차이 → attention 점수에서 ref 별 자연 분리.
        ref_pre_norm / ref_kv_proj / ref_k_norm 은 메인과 별도 weight — 조건성 강제.

        Args:
            refs:    (B, L_ref, d) 참조 이미지 토큰들 (모든 ref 합친 시퀀스).
            pe_refs: (B, 1, L_ref, D/2, 2, 2) ref RoPE (각 ref 마다 다른 t_offset).

        Returns:
            ref_k: (B, n_kv_heads, L_ref, D) — norm + RoPE, 또는 None
            ref_v: (B, n_kv_heads, L_ref, D), 또는 None
        """
        if refs is None or refs.shape[1] == 0:
            return None, None

        # self.ref_pre_norm + self.ref_kv_proj — 메인과 별도 가중치 (조건성을 weight 차원에서 강제).
        ref_kv       = self.ref_kv_proj(self.ref_pre_norm(refs))                              # (B, L_ref, 2·n_kv·head_dim)
        ref_k, ref_v = rearrange(ref_kv, "B L (K H D) -> K B H L D", K=2, H=self.n_kv_heads)  # 2 × (B, n_kv, L_ref, D)
        # self.ref_k_norm — RMSNorm(head_dim).   ref K 만 정규화 (txt 와 동일 패턴).
        ref_k        = self.ref_k_norm(ref_k)                                                 # head_dim 단위 RMSNorm
        # apply_rope: pe_refs 의 head 축은 1 broadcast — n_kv_heads 든 num_heads 든 호환.
        ref_k        = apply_rope(ref_k, pe_refs)                                             # t_offset 으로 ref 별 분리
        return ref_k, ref_v

    def attn_forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe:  Tensor,
        mod: ModulationOut,
        attn_mask: Tensor | None = None,
        refs:      Tensor | None = None,
        pe_refs:   Tensor | None = None,
    ) -> Tensor:
        """이미지·텍스트·(옵션)reference 통합 attention.

        처리 흐름:
            1. Q/K/V 생성 (메인 + RoPE / 텍스트 / ref + RoPE)  ← _compute_img_qkv / _compute_txt_kv / _compute_ref_kv
            2. K/V concat: [txt ; (ref) ; img]                   (Q 는 메인 이미지만)
            3. GQA expand: n_kv_heads → num_heads
            4. attn_mask → SDPA bool visibility mask             (텍스트 패딩만 마스킹, ref/img 항상 visible)
            5. attention:  SDPA
            6. 출력 사영

        Args:
            img:       (B, L_img, d) 메인 이미지 토큰.
            txt:       (B, L_txt, d) 텍스트 인코더 출력.
            pe:        (B, 1, L_img, D/2, 2, 2) 메인 이미지 RoPE.
            mod:       AdaLN scale/shift, 각 (B, 1, d).
            attn_mask: None 또는 (B, L_txt) 0/1 마스크.
            refs:      None 또는 (B, L_ref, d) 참조 이미지 토큰들.
            pe_refs:   None 또는 (B, 1, L_ref, D/2, 2, 2) ref RoPE.

        Returns:
            (B, L_img, d) — Q 가 메인 이미지만이라 strip 불필요.
        """
        # Step 1. 입력별 Q/K/V 생성 (메인 / 텍스트 / ref).   RoPE 는 각 빌더 안에서 적용됨.
        img_q, img_k, img_v = self._compute_img_qkv(img, pe, mod)
        txt_k, txt_v        = self._compute_txt_kv(txt)
        ref_k, ref_v        = self._compute_ref_kv(refs, pe_refs)

        # Step 2. K/V 시퀀스 차원 concat = [txt ; (ref) ; img]
        if ref_k is not None:
            k = torch.cat((txt_k, ref_k, img_k), dim=2)                                       # (B, n_kv, L_txt+L_ref+L_img, D)
            v = torch.cat((txt_v, ref_v, img_v), dim=2)
        else:
            k = torch.cat((txt_k, img_k), dim=2)                                              # (B, n_kv, L_txt+L_img, D)
            v = torch.cat((txt_v, img_v), dim=2)

        # Step 3. GQA expand — n_kv_heads → num_heads (group_size=1 이면 no-op)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)                                   # (B, num_heads, L_total, D)
            v = v.repeat_interleave(self.group_size, dim=1)                                   # (B, num_heads, L_total, D)

        # Step 4. 텍스트 패딩 마스크 → SDPA bool visibility mask (ref/img 항상 visible)
        if attn_mask is not None:
            bs, _, l_img, _ = img_q.shape
            l_txt           = txt_k.shape[2]
            l_ref           = ref_k.shape[2] if ref_k is not None else 0
            assert attn_mask.dim() == 2 and attn_mask.shape[-1] == l_txt, (
                f"attention_mask shape {attn_mask.shape} 가 (B, L_txt={l_txt}) 와 불일치"
            )
            ones_img   = torch.ones((bs, l_img), dtype=torch.bool, device=img_q.device)       # 메인 이미지 항상 visible
            mask_parts = [attn_mask.to(torch.bool)]
            if l_ref > 0:
                mask_parts.append(torch.ones((bs, l_ref), dtype=torch.bool, device=img_q.device))  # ref 항상 visible
            mask_parts.append(ones_img)
            joint_mask = torch.cat(mask_parts, dim=-1)                                        # (B, L_txt+L_ref+L_img)
            attn_mask  = joint_mask[:, None, None, :]                                      # (B, 1, 1, L_all) broadcast by SDPA

        # Step 5. attention — SDPA
        # compute_attention() — 위 정의.   SDPA wrapper (cuDNN backend 우선).
        attn = compute_attention(img_q, k, v, attn_mask=attn_mask)                            # (B, H, L_img, D)

        # Step 6. 출력 사영
        attn = rearrange(attn, "B H L D -> B L (H D)")                                        # (B, L_img, d)
        return self.attn_out(attn)                                                            # (B, L_img, d)

    def ffn_forward(
        self,
        x:   Tensor,
        mod: ModulationOut,
    ) -> Tensor:
        """GEGLU MLP.  x: (B, L_img, d) → (B, L_img, d)."""
        # self.post_attention_layernorm — LayerNorm 또는 RMSNorm (affine=False).   AdaLN-Zero scale/shift 적용.
        x = (1 + mod.scale) * self.post_attention_layernorm(x) + mod.shift                            # (B, L_img, d)
        # self.gate_proj / self.up_proj — nn.Linear(hidden → mlp_hidden_dim).   GEGLU 두 갈래.
        gate = self.gate_proj(x)                                                                       # (B, L_img, h)
        up   = self.up_proj(x)                                                                         # (B, L_img, h)
        # self.mlp_act — nn.GELU(approximate="tanh").   GEGLU (gate·GELU ⊙ up).
        gate_act = self.mlp_act(gate)                                                                  # (B, L_img, h)
        # self.down_proj — nn.Linear(mlp_hidden_dim → hidden, bias=False).   MLP 출력 차원 복원.
        return self.down_proj(gate_act * up)                                                          # (B, L_img, d)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Tensor | None = None,
        refs:    Tensor | None = None,
        pe_refs: Tensor | None = None,
        **_,
    ) -> Tensor:
        """img 잔차 두 번 (attention + MLP).  Zero gate 로 학습 시작 시 identity 보장.

        refs / pe_refs 는 참조 이미지 입력 (선택).   None 이면 표준 T2I 동작.
        """
        # self.modulation(Modulation) — vec(B, embed_dim) → (mod_attn, mod_mlp) 두 path.
        #   각 ModulationOut: shift/scale/gate (각 (B, 1, d)).   Zero-init 으로 초기엔 identity.
        mod_attn, mod_mlp = self.modulation(vec)                                                      # 각 ModulationOut: (B, 1, d) × 3
        # Sandwich-Norm: sandwich_norm=False 면 attn_post_norm/ffn_post_norm 이 Identity.
        # self.attn_forward — 위 메서드.   AdaLN + GQA Q/K/V + MMA.
        attn_out = self.attn_forward(
            img, txt, pe, mod_attn,
            attn_mask=attention_mask,
            refs=refs, pe_refs=pe_refs,
        )                                                                                              # (B, L_img, d)
        # self.attn_post_norm — Sandwich-Norm ON 시 RMSNorm, OFF 시 Identity.   잔차 합산 전 한 번 더 정규화.
        img = img + mod_attn.gate * self.attn_post_norm(attn_out)
        # self.ffn_forward — 위 메서드.   AdaLN + GEGLU.
        # self.ffn_post_norm — Sandwich-Norm 패턴, attn_post_norm 과 동일 패턴.
        img = img + mod_mlp.gate  * self.ffn_post_norm(self.ffn_forward(img, mod_mlp))                # (B, L_img, d)
        return img


class PIERROTDualBlock(nn.Module):
    """텍스트도 Q 를 만들고 자체 MLP / AdaLN 을 갖는 양방향 블록.

    동기:
        표준 PIERROTBlock 에서 텍스트는 KV-only 라 블록을 거쳐도 업데이트되지 않는다.
        텍스트도 Q 까지 만들어 joint attention 에 참여시키고, 텍스트 전용 MLP 로 정제 →
        양방향 흐름 회복.   초반 cross-modal 정합 단계에서만 의미 크므로 앞 N 블록만 권장.

    구성:
        - 이미지 측: PIERROTBlock 과 동일 (LN + QKV + attn_out + MLP + AdaLN)
        - 텍스트 측: 거울 (LN + QKV + attn_out + MLP + AdaLN — 이미지와 별도 가중치)
        - 참조 측: PIERROTBlock 의 ref_kv_proj 그대로 유지 (Q 없음, KV-only)

    Forward 인자 (PIERROTBlock 과 호환):
        img : (B, L_img, hidden_size)
        txt : (B, L_txt, hidden_size)
        vec : (B, hidden_size)                  timestep embedding
        pe  : (B, 1, L_img, D/2, 2, 2)          이미지 RoPE
        attention_mask : None 또는 (B, L_txt)
        refs           : None 또는 (B, L_ref, hidden)
        pe_refs        : None 또는 (B, 1, L_ref, D/2, 2, 2)

    반환:
        (img, txt) — 둘 다 업데이트.   PIERROTBlock 이 img 만 반환하는 것과 다름.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        sandwich_norm: bool = False,
        use_tanh_gate: bool = False,
        adaln_4param: bool = False,
        use_rmsnorm: bool = False,
        adaln_input_dim: int | None = None,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.hidden_size     = hidden_size
        self.num_heads       = num_heads
        # K/V head 수 — None=num_heads (MHA), int=GQA.
        self.n_kv_heads      = n_kv_heads if n_kv_heads is not None else num_heads
        self.group_size      = num_heads // self.n_kv_heads
        self.head_dim        = hidden_size // num_heads
        self.scale           = qk_scale or self.head_dim ** -0.5
        self.mlp_hidden_dim  = int(hidden_size * mlp_ratio)
        self.sandwich_norm   = sandwich_norm
        self.use_tanh_gate   = use_tanh_gate
        self.adaln_4param    = adaln_4param
        self.use_rmsnorm     = use_rmsnorm
        self.adaln_input_dim = adaln_input_dim

        # 공유 — Q, K 정규화 (이미지·텍스트 모두 통과)
        self.qk_norm = QKNorm(self.head_dim)

        # GQA 차원 계산
        qkv_out_dim          = self.head_dim * (num_heads + 2 * self.n_kv_heads)   # Q n_heads + K/V n_kv_heads × 2
        kv_out_dim           = self.head_dim * 2 * self.n_kv_heads                 # K, V 만

        # 이미지 측 (PIERROTBlock 과 동일 구조)
        self.img_pre_norm   = _make_block_norm(hidden_size, use_rmsnorm)
        self.img_qkv_proj   = nn.Linear(hidden_size, qkv_out_dim, bias=False)
        self.img_attn_out   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.img_post_norm  = _make_block_norm(hidden_size, use_rmsnorm)
        self.img_gate_proj  = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.img_up_proj    = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.img_down_proj  = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.img_modulation = Modulation(
            hidden_size,
            use_tanh_gate=use_tanh_gate,
            adaln_4param=adaln_4param,
            vec_dim=adaln_input_dim,
        )

        # 텍스트 측 (거울 — 이게 핵심 추가)
        # PIERROTBlock 은 txt 가 KV-only 였는데 여기서는 txt 도 Q 까지 + 자체 MLP + 자체 AdaLN.
        self.txt_pre_norm   = _make_block_norm(hidden_size, use_rmsnorm)
        self.txt_qkv_proj   = nn.Linear(hidden_size, qkv_out_dim, bias=False)                  # ← Q 까지 (GQA 적용)
        self.txt_attn_out   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.txt_post_norm  = _make_block_norm(hidden_size, use_rmsnorm)
        self.txt_gate_proj  = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.txt_up_proj    = nn.Linear(hidden_size, self.mlp_hidden_dim, bias=False)
        self.txt_down_proj  = nn.Linear(self.mlp_hidden_dim, hidden_size, bias=False)
        self.txt_modulation = Modulation(
            hidden_size,
            use_tanh_gate=use_tanh_gate,
            adaln_4param=adaln_4param,
            vec_dim=adaln_input_dim,
        )

        # 참조(reference) 이미지 K/V — PIERROTBlock 과 동일 (Q 없음)
        self.ref_pre_norm = _make_block_norm(hidden_size, use_rmsnorm)
        self.ref_kv_proj  = nn.Linear(hidden_size, kv_out_dim, bias=False)
        self.ref_k_norm   = RMSNorm(self.head_dim)                                          # affine=True 학습

        # 활성 (이미지·텍스트 MLP 공유)
        self.mlp_act = nn.GELU(approximate="tanh")

        # Sandwich-Norm
        # img / txt 두 stream 각각 attn 출력·MLP 출력에 Post-Norm.   sandwich_norm=False 면 Identity.
        self.img_attn_post_norm = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()
        self.img_ffn_post_norm  = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()
        self.txt_attn_post_norm = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()
        self.txt_ffn_post_norm  = RMSNorm(hidden_size) if sandwich_norm else nn.Identity()

    def _img_mlp(
        self,
        x:   Tensor,
        mod: ModulationOut,
    ) -> Tensor:
        """이미지 GEGLU MLP."""
        x    = (1 + mod.scale) * self.img_post_norm(x) + mod.shift
        gate = self.img_gate_proj(x)
        up   = self.img_up_proj(x)
        gate_act = self.mlp_act(gate)                                                           # GELU
        return self.img_down_proj(gate_act * up)

    def _txt_mlp(self, x: Tensor, mod: ModulationOut) -> Tensor:
        """텍스트 GEGLU MLP — 이미지와 별도 가중치."""
        # self.txt_post_norm — LayerNorm/RMSNorm(affine=False).   AdaLN-Zero scale/shift 적용.
        x = (1 + mod.scale) * self.txt_post_norm(x) + mod.shift
        # self.txt_gate_proj / txt_up_proj / txt_down_proj — 각각 nn.Linear, img 측과 별도 가중치.
        # self.mlp_act(GELU tanh) — gate 활성화.   gate·GELU ⊙ up → down_proj.
        return self.txt_down_proj(self.mlp_act(self.txt_gate_proj(x)) * self.txt_up_proj(x))

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Tensor | None = None,
        refs:    Tensor | None = None,
        pe_refs: Tensor | None = None,
        **_,
    ) -> tuple[Tensor, Tensor]:
        """img, txt 둘 다 업데이트해서 반환."""
        # AdaLN 변조 (이미지·텍스트 각각).
        # self.img_modulation / self.txt_modulation(Modulation) — 별도 학습 가중치 (img/txt 분리).
        #   각자 (mod_attn, mod_mlp) 두 ModulationOut 반환.
        img_mod_a, img_mod_m = self.img_modulation(vec)
        txt_mod_a, txt_mod_m = self.txt_modulation(vec)

        # GQA: Q 풀 사이즈, K/V 작은 사이즈로 분리
        q_dim    = self.head_dim * self.num_heads
        kv_dim_1 = self.head_dim * self.n_kv_heads

        # 이미지 Q/K/V (AdaLN + RoPE)
        # self.img_pre_norm — affine-free LayerNorm/RMSNorm.   AdaLN-Zero scale/shift 적용.
        img_x   = (1 + img_mod_a.scale) * self.img_pre_norm(img) + img_mod_a.shift            # (B, L_img, d)
        # self.img_qkv_proj — Linear(hidden → head_dim·(n+2·n_kv)).   Q/K/V 한 번에 (속도/메모리 효율).
        img_qkv = self.img_qkv_proj(img_x)                                                    # (B, L_img, head_dim·(n+2·n_kv))
        # tensor.split — Q/K/V 비율로 3 분할 후 einops.rearrange 로 head 축 분리.
        img_q_flat, img_k_flat, img_v_flat = img_qkv.split([q_dim, kv_dim_1, kv_dim_1], dim=-1)
        img_q   = rearrange(img_q_flat, "B L (H D) -> B H L D", H=self.num_heads)             # Q 풀
        img_k   = rearrange(img_k_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)            # K 작음
        img_v   = rearrange(img_v_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)
        # self.qk_norm(QKNorm) — Q/K 각자 RMSNorm.   bf16 안정성.
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)
        # apply_rope() — 4D RoPE 회전.   pe 의 head 축 1 → num_heads 로 자동 broadcast.
        img_q = apply_rope(img_q, pe)
        img_k = apply_rope(img_k, pe)

        # 텍스트 Q/K/V (AdaLN, RoPE 미적용 — 인코더 출력 신뢰)
        # self.txt_pre_norm + self.txt_qkv_proj — img 측과 동일 구조의 거울 (별도 학습 가중치).
        txt_x   = (1 + txt_mod_a.scale) * self.txt_pre_norm(txt) + txt_mod_a.shift            # (B, L_txt, d)
        txt_qkv = self.txt_qkv_proj(txt_x)                                                    # (B, L_txt, head_dim·(n+2·n_kv))
        txt_q_flat, txt_k_flat, txt_v_flat = txt_qkv.split([q_dim, kv_dim_1, kv_dim_1], dim=-1)
        txt_q   = rearrange(txt_q_flat, "B L (H D) -> B H L D", H=self.num_heads)
        txt_k   = rearrange(txt_k_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)
        txt_v   = rearrange(txt_v_flat, "B L (H D) -> B H L D", H=self.n_kv_heads)
        # txt 도 같은 self.qk_norm 통과 — QKNorm 인스턴스는 img/txt 공유 (head_dim 단위 정규화라 무관).
        txt_q, txt_k = self.qk_norm(txt_q, txt_k, txt_v)

        # 참조 K/V (있을 때만)
        ref_k = ref_v = None
        if refs is not None and refs.shape[1] > 0:
            # self.ref_pre_norm + self.ref_kv_proj — 메인과 별도 가중치 (Q 없음).
            ref_kv       = self.ref_kv_proj(self.ref_pre_norm(refs))                          # (B, L_ref, 2·n_kv·head_dim)
            ref_k, ref_v = rearrange(ref_kv, "B L (K H D) -> K B H L D", K=2, H=self.n_kv_heads)
            # self.ref_k_norm(RMSNorm) — ref K 만 정규화 (V 는 정규화 안 함).
            ref_k        = self.ref_k_norm(ref_k)
            # apply_rope() — ref 별 t_offset 으로 회전 차이 → attention 자연 분리.
            ref_k        = apply_rope(ref_k, pe_refs)

        # Joint attention — Q 도 합침 (PIERROTBlock 과의 핵심 차이)
        # K, V 순서: [txt ; (ref) ; img]   Q 순서: [txt ; img]
        Q = torch.cat([txt_q, img_q], dim=2)                                                  # (B, num_heads, L_txt + L_img, D)
        if ref_k is not None:
            K = torch.cat([txt_k, ref_k, img_k], dim=2)                                       # (B, n_kv, L_total, D)
            V = torch.cat([txt_v, ref_v, img_v], dim=2)
        else:
            K = torch.cat([txt_k, img_k], dim=2)                                              # (B, n_kv, L_total, D)
            V = torch.cat([txt_v, img_v], dim=2)

        # GQA expand: K/V (n_kv_heads) → num_heads
        if self.group_size > 1:
            K = K.repeat_interleave(self.group_size, dim=1)                                   # (B, num_heads, L_total, D)
            V = V.repeat_interleave(self.group_size, dim=1)

        # Attention mask 확장 (txt 패딩만 마스킹, ref/img 는 항상 visible)
        # joint attention 에선 Q 길이 = L_txt + L_img 이므로 마스크도 그 크기로
        attn_mask_4d = None
        if attention_mask is not None:
            bs    = img_q.shape[0]
            l_img = img_q.shape[2]
            l_txt = txt_q.shape[2]
            l_ref = ref_k.shape[2] if ref_k is not None else 0

            # K 마스크: [txt 패딩, ref 모두 보임, img 모두 보임]
            ones_img = torch.ones((bs, l_img), dtype=torch.bool, device=img_q.device)
            mask_parts = [attention_mask.to(torch.bool)]
            if l_ref > 0:
                mask_parts.append(torch.ones((bs, l_ref), dtype=torch.bool, device=img_q.device))
            mask_parts.append(ones_img)
            joint_mask = torch.cat(mask_parts, dim=-1)                                        # (B, L_kv)

            # Q 길이 = L_txt + L_img.   각 query 토큰이 위 K 들을 볼 때 같은 마스크.
            l_q = l_txt + l_img
            attn_mask_4d = joint_mask[:, None, None, :]

        # Attention 실행
        # compute_attention — 위 정의.   cuDNN SDPA wrapper.
        attn = compute_attention(Q, K, V, attn_mask=attn_mask_4d)                              # (B, H, L_txt + L_img, D)

        # 출력 분리: [txt 부분, img 부분]
        # tensor 슬라이싱 attn[:, :, :l_txt, :] / [:, :, l_txt:, :] — L 축에서 txt/img 분리.
        l_txt    = txt_q.shape[2]
        # rearrange "B H L D -> B L (H D)" — head 들을 hidden 축으로 합치는 inverse.
        attn_txt = rearrange(attn[:, :, :l_txt, :], "B H L D -> B L (H D)")                   # (B, L_txt, d)
        attn_img = rearrange(attn[:, :, l_txt:, :], "B H L D -> B L (H D)")                   # (B, L_img, d)
        # self.txt_attn_out / self.img_attn_out — 각자 별도 nn.Linear (img/txt 가중치 분리).
        attn_txt = self.txt_attn_out(attn_txt)
        attn_img = self.img_attn_out(attn_img)

        # 잔차 (각각 자기 gate 로) — Sandwich-Norm 시 잔차 항이 RMSNorm 통과
        # self.img_attn_post_norm / self.txt_attn_post_norm — Sandwich-Norm ON 시 RMSNorm, OFF 시 Identity.
        img = img + img_mod_a.gate * self.img_attn_post_norm(attn_img)                        # (B, L_img, d)
        txt = txt + txt_mod_a.gate * self.txt_attn_post_norm(attn_txt)                        # (B, L_txt, d)

        # MLP 잔차 (각각 별도 MLP + 별도 gate)
        # self._img_mlp — 위 메서드.   GEGLU.
        # self.img_ffn_post_norm — Sandwich-Norm 패턴.
        img = img + img_mod_m.gate * self.img_ffn_post_norm(self._img_mlp(img, img_mod_m))     # (B, L_img, d)
        # self._txt_mlp — 위 메서드.   GEGLU (img 측과 별도 가중치).
        txt = txt + txt_mod_m.gate * self.txt_ffn_post_norm(self._txt_mlp(txt, txt_mod_m))    # (B, L_txt, d)

        return img, txt   # 둘 다 반환 — PIERROTBlock 과의 결정적 차이


class LastLayer(nn.Module):
    """디노이저 최종 projection.  AdaLN(2-param, gate 없음) + Zero init Linear.

    옵션:
        use_rmsnorm     : norm_final 을 LayerNorm(affine=False) 대신 RMSNorm(affine=False) 로.
                          False (기본) — LayerNorm.   True — RMSNorm.
        adaln_input_dim : adaLN_modulation Linear 의 입력 차원 (vec_dim).
                          None (기본) — hidden_size.
                          int        — vec_dim → 2·hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False,
        adaln_input_dim: int | None = None,
    ):
        super().__init__()
        # pre-final norm 도 _make_block_norm 으로 분기.   둘 다 affine 없음.
        self.norm_final       = _make_block_norm(hidden_size, use_rmsnorm)
        self.linear           = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        # adaLN_modulation 의 Linear 입력 차원 = vec_dim (None=hidden_size).
        adaln_in = adaln_input_dim if adaln_input_dim is not None else hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_in, 2 * hidden_size, bias=True),                                          # vec_dim → (shift, scale).  gate 없음
        )
        # Zero init — 학습 시작 시 모델 출력이 0 → Flow Matching 의 안전한 출발점
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """x: (B, N, d), vec: (B, d) → (B, N, p*p*out_channels)."""
        # self.adaLN_modulation — Sequential(SiLU → Linear(adaln_in → 2·d)).   shift/scale 두 chunk.
        # tensor.chunk(2, dim=1) — 채널 축에서 정확히 절반씩 → (shift, scale) 두 텐서.
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)                                     # 각 (B, d)
        # self.norm_final — LayerNorm 또는 RMSNorm (affine=False).   AdaLN 의 (1+scale)·LN(x)+shift 패턴.
        #   shift[:, None, :] / scale[:, None, :] — (B, d) → (B, 1, d) broadcast 를 위한 axis 추가.
        x            = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]               # (B, N, d)
        # self.linear — Zero-init Linear(hidden → p·p·out_ch).   학습 시작 시 출력 0 (Flow Matching 안전).
        return self.linear(x)                                                                         # (B, N, p*p*out_channels)
