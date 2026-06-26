"""PIERROT 디퓨전 트랜스포머 본체.

구성:
    PIERROTParams        — 하이퍼파라미터 dataclass
    img2seq / seq2img    — 패치화 / 역패치화
    PIERROT              — 모델 클래스 (forward, 4D RoPE, multi-ref 지원)
"""
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.functional import fold, unfold

from .pierrot_modules import (
    EmbedND,  # spellchecker:disable-line
    LastLayer,
    MLPEmbedder,
    PIERROTBlock,
    PIERROTDualBlock,
    TextAdapter,
    get_image_ids_4d,
    get_text_ids_4d,
    timestep_embedding,
)


@dataclass
class PIERROTParams:
    """PIERROT 모델 하이퍼파라미터.   dict 로도 받을 수 있다 (`PIERROT(dict(...))`).

    제약:
        - hidden_size 는 num_heads 로 나누어떨어져야 함
        - len(axes_dim) == 4 (4D RoPE 전용)
        - sum(axes_dim) == hidden_size / num_heads (= head_dim)
    """
    in_channels:    int                # latent 채널 수
    patch_size:     int                # 패치 크기 (patch_size=2 → H/2, W/2 토큰)
    context_in_dim: int                # 텍스트 인코더 출력 차원
    hidden_size:    int                # 트랜스포머 은닉 차원
    mlp_ratio:      float              # MLP 확장 비율
    num_heads:      int                # attention head 수
    depth:          int                # 트랜스포머 블록 개수
    axes_dim:       list[int]          # 4축 (t, h, w, l) 채널 분배
    theta:          int                # RoPE 주파수 base

    time_factor:     float = 1000.0    # timestep 스케일 팩터
    time_max_period: int   = 10_000    # sinusoidal 주파수 스펙트럼 하한

    conditioning_block_ids: list[int] | None = None   # 일부 블록만 conditioning block 으로 쓸 때
    bottleneck_size:        int       | None = None   # None=단일 Linear, 정수=2-layer bottleneck
    dual_block_count:       int             = 0       # 앞 N 개 블록만 PIERROTDualBlock
    sandwich_norm:          bool            = False   # Z-Image: attn/MLP 출력에 추가 RMSNorm
    use_tanh_gate:          bool            = False   # Z-Image: AdaLN gate 에 tanh 적용 (±1 클램프)
    adaln_4param:           bool            = False   # Z-Image: 4-param AdaLN (shift 제거)
    n_kv_heads:             int  | None     = None    # Z-Image: GQA — K/V head 수, None=MHA
    adaln_embed_dim:        int  | None     = None    # Z-Image: modulation Linear 입력 차원 cap
    use_rmsnorm:            bool            = False   # Z-Image: 블록 pre/post norm 을 RMSNorm 으로
    axes_max_len:           list[int] | None = None   # Z-Image: RoPE freqs precompute cache 각 축 max
    use_text_adapter:       bool            = False   # i1: txt_in 뒤 self-attention adapter (zero-init 잔차, OFF=항등)
    text_adapter_depth:     int             = 2       # text adapter transformer 블록 수
    text_adapter_mlp_ratio: float           = 4.0     # text adapter MLP 확장 비율


def img2seq(img: Tensor, patch_size: int) -> Tensor:
    """이미지 latent → 패치 시퀀스.

    Args:
        img: (B, C, H, W).   H, W 는 patch_size 의 배수.
        patch_size: 패치 크기.

    Returns:
        (B, N, C * p * p).   N = (H/p) * (W/p).
    """
    # torch.nn.functional.unfold — sliding window 로 (kernel_size × kernel_size) 패치를 마지막 dim 으로 모음.
    #   stride=patch_size 라 겹침 없음.   결과 shape: (B, C·p·p, N)   N = (H/p)·(W/p).
    # transpose(1, 2) — (B, C·p·p, N) → (B, N, C·p·p) 로 axis swap. Transformer 의 (B, L, D) 컨벤션에 맞춤.
    return unfold(img, kernel_size=patch_size, stride=patch_size).transpose(1, 2)


def seq2img(seq: Tensor, patch_size: int, shape) -> Tensor:
    """`img2seq` 의 역변환.   패치 시퀀스 → 이미지 latent.

    Args:
        seq: (B, N, C * p * p).
        patch_size: 패치 크기.
        shape: 목표 (H, W).   tuple 또는 Tensor.

    Returns:
        (B, C, H, W).
    """
    if isinstance(shape, tuple):
        shape = shape[-2:]
    elif isinstance(shape, torch.Tensor):
        shape = (int(shape[0]), int(shape[1]))
    else:
        raise NotImplementedError(f"shape type {type(shape)} not supported")
    # transpose(1, 2) — (B, N, C·p·p) → (B, C·p·p, N).   fold() 가 기대하는 dim 순서로 swap.
    # torch.nn.functional.fold — unfold 역연산.   stride=patch_size + 겹침 없음 → 패치들을 (H, W) 격자로 짜맞춤.
    #   결과 shape: (B, C, H, W).   C = (C·p·p) / (p·p) (fold 자동 계산).
    return fold(seq.transpose(1, 2), shape, kernel_size=patch_size, stride=patch_size)


class PIERROT(nn.Module):
    """PIERROT 디퓨전 트랜스포머.

    호출처:   training/train.py 의 main 에서 build (config_size + Z-Image 옵션 dict 주입).
              scripts/sample.py 의 추론 entry 에서 build.
    forward 입력:   (image_latent, timestep, prompt_embeds, mask, ref_latents, ...) — 학습/추론 공통.
    forward 출력:   (B, C, H, W) — Flow Matching velocity 또는 x_prediction.

    동작 흐름:
        image_latent → patchify → img_in → [block × depth] → final_layer → unpatchify

    구조:
        - 앞 dual_block_count 개 = PIERROTDualBlock (text 도 Q 생성·정제, MMDiT 스타일)
        - 나머지 depth - dual_block_count = PIERROTBlock (text KV-only, 비대칭)
        - 4D RoPE (t, h, w, l): 메인 이미지 + 다중 참조 + 어순까지 한 좌표계로 통합
        - Z-Image 토글 옵션 결합 (PIERROTParams 필드 참조)
    """

    transformer_block_class = PIERROTBlock

    def _init_params(self, params: PIERROTParams) -> None:
        """PIERROTParams 의 주요 필드를 인스턴스 속성으로 복사."""
        self.params          = params
        self.in_channels     = params.in_channels
        self.patch_size      = params.patch_size
        self.out_channels    = self.in_channels * self.patch_size ** 2   # 한 패치의 원소 수
        self.time_factor     = params.time_factor
        self.time_max_period = params.time_max_period
        self.hidden_size     = params.hidden_size
        self.num_heads       = params.num_heads

    def __init__(self, params: PIERROTParams | dict[str, Any] | None = None, **kwargs: Any):
        """
        Args:
            params: PIERROTParams 또는 dict.   None 이면 **kwargs 로 구성 (체크포인트 로드 경로).
            kwargs: params 가 None 일 때 PIERROTParams 필드를 직접 받음.
        """
        super().__init__()

        # params 정규화 — dict / kwargs / PIERROTParams 모두 허용
        if params is None:
            params = kwargs
        if isinstance(params, dict):
            params = PIERROTParams(**{k: v for k, v in params.items() if not k.startswith("_")})
        elif not isinstance(params, PIERROTParams):
            raise TypeError(f"params must be PIERROTParams or dict, got {type(params)}")

        self._init_params(params)

        # 차원 검증
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"hidden_size {params.hidden_size} 가 num_heads {params.num_heads} 로 나누어떨어지지 않음")

        pe_dim = params.hidden_size // params.num_heads   # head_dim

        if len(params.axes_dim) != 4:
            raise ValueError(
                f"PIERROT 는 4D RoPE 만 사용한다.  axes_dim 길이는 4 여야 한다 (t, h, w, l).  "
                f"현재 axes_dim={params.axes_dim} (length {len(params.axes_dim)})."
            )
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"sum(axes_dim) {sum(params.axes_dim)} != head_dim {pe_dim}")

        if params.axes_max_len is not None and len(params.axes_max_len) != len(params.axes_dim):
            raise ValueError(
                f"axes_max_len 길이 {len(params.axes_max_len)} 가 axes_dim 길이 "
                f"{len(params.axes_dim)} 와 일치해야 함"
            )

        # GQA 검증 — n_kv_heads 가 None 이면 MHA (n_kv_heads = num_heads)
        if params.n_kv_heads is not None:
            if params.n_kv_heads <= 0:
                raise ValueError(f"n_kv_heads 는 양수여야 함 (받은 값 {params.n_kv_heads})")
            if params.n_kv_heads > params.num_heads:
                raise ValueError(
                    f"n_kv_heads {params.n_kv_heads} 가 num_heads {params.num_heads} 보다 클 수 없음"
                )
            if params.num_heads % params.n_kv_heads != 0:
                raise ValueError(
                    f"n_kv_heads {params.n_kv_heads} 가 num_heads {params.num_heads} 의 약수여야 함"
                )

        # RoPE 임베더 — ids:(B, N, 4) → pe:(B, 1, N, head_dim/2, 2, 2)
        self.pe_embedder = EmbedND(  # spellchecker:disable-line
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim,
            axes_max_len=params.axes_max_len,
        )

        # 이미지 입력 projection
        patch_dim = self.in_channels * self.patch_size ** 2
        if params.bottleneck_size is not None:
            self.img_in = nn.Sequential(
                nn.Linear(patch_dim, params.bottleneck_size, bias=True),         # (B, N, patch_dim)  → (B, N, bottleneck)
                nn.Linear(params.bottleneck_size, self.hidden_size, bias=True),  # (B, N, bottleneck) → (B, N, hidden)
            )
        else:
            self.img_in = nn.Linear(patch_dim, self.hidden_size, bias=True)      # (B, N, patch_dim) → (B, N, hidden)

        # modulation Linear 입력 차원 cap — adaln_embed_dim 가 None 이면 hidden_size 그대로
        self.effective_embed_dim = (
            min(self.hidden_size, params.adaln_embed_dim) if params.adaln_embed_dim is not None
            else self.hidden_size
        )

        # timestep / 텍스트 사영
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.effective_embed_dim)  # (B, 256)              → (B, effective_embed_dim)
        self.txt_in  = nn.Linear(params.context_in_dim, self.hidden_size)            # (B, L_txt, ctx_dim)   → (B, L_txt, hidden)

        # txt_in 뒤 self-attention adapter (옵션) — zero-init 잔차라 OFF/어댑터 없는 ckpt 와 출력 동일.
        self.text_adapter = (
            TextAdapter(
                dim       = self.hidden_size,
                depth     = params.text_adapter_depth,
                num_heads = self.num_heads,
                mlp_ratio = params.text_adapter_mlp_ratio,
            )
            if params.use_text_adapter else None
        )

        # 트랜스포머 스택 — 앞 dual_block_count 개는 PIERROTDualBlock, 나머지는 PIERROTBlock
        conditioning_block_ids = params.conditioning_block_ids or list(range(params.depth))

        def block_class(idx: int) -> type:
            if idx < params.dual_block_count:
                return PIERROTDualBlock
            return self.transformer_block_class if idx in conditioning_block_ids else PIERROTBlock

        self.blocks = nn.ModuleList([
            block_class(i)(
                self.hidden_size, self.num_heads,
                mlp_ratio             = params.mlp_ratio,
                sandwich_norm         = params.sandwich_norm,
                use_tanh_gate         = params.use_tanh_gate,
                adaln_4param          = params.adaln_4param,
                use_rmsnorm           = params.use_rmsnorm,
                adaln_input_dim       = self.effective_embed_dim,
                n_kv_heads            = params.n_kv_heads,
            )
            for i in range(params.depth)
        ])

        # 최종 projection — hidden → patch_size^2 * out_channels (unpatchify 직전 차원)
        self.final_layer = LastLayer(
            self.hidden_size, 1, self.out_channels,
            use_rmsnorm     = params.use_rmsnorm,
            adaln_input_dim = self.effective_embed_dim,
        )

    def process_inputs(self, image_latent: Tensor, txt: Tensor, **_: Any) -> tuple[Tensor, Tensor, Tensor]:
        """패치화·텍스트 투영·RoPE 계산.

        Args:
            image_latent: (B, C, H, W).
            txt: (B, L_txt, context_in_dim).

        Returns:
            img: (B, N, C * p * p)              패치화된 이미지.
            txt: (B, L_txt, hidden)             hidden 에 사영된 텍스트.
            pe:  (B, 1, N, head_dim/2, 2, 2)    메인 이미지 토큰의 RoPE.
        """
        # self.txt_in — nn.Linear(context_in_dim → hidden_size).   텍스트 인코더 출력 차원을 모델 hidden 으로 사영.
        txt         = self.txt_in(txt)                        # (B, L_txt, ctx_dim) → (B, L_txt, hidden)
        # self.text_adapter — TextAdapter (옵션).   None 이면 SKIP (txt_in 출력 그대로).
        if self.text_adapter is not None:
            txt     = self.text_adapter(txt)                 # (B, L_txt, hidden)  → (B, L_txt, hidden)
        # img2seq() — 위 정의.   unfold + transpose 로 (B, C, H, W) 를 (B, N, C·p·p) 패치 시퀀스로 펼침.
        img         = img2seq(image_latent, self.patch_size)  # (B, N, patch_dim)   N = (H/p)·(W/p)
        bs, _, h, w = image_latent.shape                      # latent H, W (patch_size 의 배수 가정)

        # 메인 이미지 4D 좌표 — t=0 (image token), h/w=patch grid, l=0 (dummy)
        # ref 좌표는 _encode_refs_pre_in 에서 t_offset=10/20/... 으로 분리해 생성한다.
        # get_image_ids_4d() — pierrot_modules.py 정의.   (B, N, 4) 4D ids 생성, t 축이 메인/ref 분리 신호.
        img_ids = get_image_ids_4d(bs, h, w, patch_size=self.patch_size, device=image_latent.device)  # (B, N, 4)
        # self.pe_embedder — EmbedND (4D RoPE).   ids → (B, 1, N, head_dim/2, 2, 2) 회전행렬 텐서.
        #   head 축이 1 인 이유 — apply_rope 에서 num_heads 로 broadcast 됨.
        pe      = self.pe_embedder(img_ids)                                                            # (B, 1, N, head_dim/2, 2, 2)
        return img, txt, pe

    def compute_timestep_embedding(self, timestep: Tensor, dtype: torch.dtype) -> Tensor:
        """연속 timestep → hidden 차원 벡터.

        Args:
            timestep: (B,).
            dtype: 출력 dtype.

        Returns:
            (B, effective_embed_dim).
        """
        # timestep_embedding() — pierrot_modules.py 정의.   연속 t (예: 0.0~1.0) 를 sinusoidal 주파수
        #   스펙트럼 (cos/sin) 로 펼친다.   time_factor=1000 으로 dynamic range 확장, max_period 가 최저 주파수 결정.
        sin_emb = timestep_embedding(
            t=timestep, dim=256,
            max_period=self.time_max_period, time_factor=self.time_factor,
        ).to(dtype)                                                                # (B, 256) fp32 → 모델 dtype
        # self.time_in — MLPEmbedder (Linear → SiLU → Linear).   256 차원을 effective_embed_dim 으로 사영.
        return self.time_in(sin_emb)                                               # (B, effective_embed_dim)

    def forward_transformers(
        self,
        image_latent:   Tensor,
        prompt_embeds:  Tensor,
        timestep:       Tensor | None = None,
        time_embedding: Tensor | None = None,
        attention_mask: Tensor | None = None,
        refs:           Tensor | None = None,
        pe_refs:        Tensor | None = None,
        **block_kwargs: Any,
    ) -> Tensor:
        """트랜스포머 스택 + 최종 projection.   TREAD 같은 토큰 라우팅 wrapper 는 이 메서드를 감싸 block_kwargs 로 인자 주입.

        Args:
            image_latent:   (B, N, patch_dim).   이미 img2seq 된 상태.
            prompt_embeds:  (B, L_txt, hidden).   이미 txt_in 된 상태.
            timestep:       (B,) 또는 None.   timestep 또는 time_embedding 중 하나 필수.
            time_embedding: (B, effective_embed_dim) 또는 None.
            attention_mask: None 또는 (B, L_txt).   텍스트 패딩 마스크.
            refs:           None 또는 (B, L_ref, patch_dim).   참조 이미지 patch 시퀀스.
            pe_refs:        None 또는 (B, 1, L_ref, D/2, 2, 2).   ref RoPE.
            block_kwargs:   블록에 전달될 추가 인자 (TREAD 등).

        Returns:
            (B, N, patch_size^2 * out_channels).
        """
        # self.img_in — nn.Linear (또는 2-layer bottleneck).   패치 raw (patch_dim) → 모델 hidden.
        img = self.img_in(image_latent)                                            # (B, N, hidden)

        # ref 도 같은 img_in 통과 — 블록의 ref_kv_proj 가 그 다음 별도 처리.
        # weight 공유는 latent 통계 매칭 (메인/ref 같은 VAE), block 안 ref_kv_proj 가 condition 표현 분리.
        refs_proj = self.img_in(refs) if refs is not None else None                # (B, L_ref, hidden) | None

        # vec 결정 — time_embedding 직접 받거나 timestep 으로부터 계산
        if time_embedding is not None:
            vec = time_embedding                                                   # (B, effective_embed_dim)
        else:
            if timestep is None:
                raise ValueError("timestep 또는 time_embedding 중 하나는 반드시 제공해야 한다")
            # self.compute_timestep_embedding — 위 메서드.   sinusoidal(256) → MLPEmbedder → (B, effective_embed_dim).
            vec = self.compute_timestep_embedding(timestep, dtype=img.dtype)       # (B, effective_embed_dim)

        # 트랜스포머 스택 — PIERROTDualBlock 은 (img, txt) 둘 다 반환, PIERROTBlock 은 img 만
        for block in self.blocks:
            # block.forward — PIERROTBlock(비대칭) 또는 PIERROTDualBlock(joint Q).
            #   block_kwargs 로 pe (이미지 RoPE) + tread_step 등 wrapper 추가 인자 전달.
            #   attention_mask: (B, L_txt) 0/1 → 블록 안에서 additive bias 로 변환.
            out = block(
                img=img, txt=prompt_embeds, vec=vec,
                attention_mask=attention_mask,
                refs=refs_proj, pe_refs=pe_refs,
                **block_kwargs,                                                    # pe 는 block_kwargs 로 들어옴
            )
            if isinstance(out, tuple):
                img, prompt_embeds = out                                           # PIERROTDualBlock — txt 도 갱신
            else:
                img = out                                                          # PIERROTBlock      — txt 그대로

        # self.final_layer — LastLayer (AdaLN 2-param + zero-init Linear).
        #   hidden → patch_size²·out_channels.   학습 시작 시 0 출력 → Flow Matching 의 안전한 시작점.
        return self.final_layer(img, vec)                                          # (B, N, p*p*out_channels)

    def _encode_refs_pre_in(
        self,
        ref_latents:   list[Tensor],
        ref_t_offsets: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """다중 참조 이미지를 img2seq + 4D ids 까지 처리 (img_in 은 forward_transformers 가 일괄 적용).

        각 참조 이미지에 t 축 오프셋을 다르게 부여 → RoPE 회전 단계에서 토큰 출처가 분리됨.

        Args:
            ref_latents:   list of (B, C, H_ref_i, W_ref_i).
            ref_t_offsets: None 또는 list[int].   None 이면 자동 [10, 20, 30, ...].

        Returns:
            full_ref_seq: (B, N_ref_total, patch_dim).
            ref_pe:       (B, 1, N_ref_total, head_dim/2, 2, 2).
        """
        assert len(self.params.axes_dim) == 4, f"axes_dim 길이 4 필수, 현재 {self.params.axes_dim}"

        if ref_t_offsets is None:
            ref_t_offsets = [10 * (i + 1) for i in range(len(ref_latents))]
        elif len(ref_t_offsets) != len(ref_latents):
            raise ValueError(
                f"ref_t_offsets 길이 ({len(ref_t_offsets)}) 가 ref_latents 길이 ({len(ref_latents)}) 와 일치해야 한다"
            )

        # 각 ref 를 (img2seq + 4D ids) 로 변환
        ref_seqs:    list[Tensor] = []
        ref_ids_all: list[Tensor] = []
        for ref_latent, t_offset in zip(ref_latents, ref_t_offsets):
            bs, _, h, w = ref_latent.shape
            # img2seq() — 위 정의.   ref 도 메인과 같은 패치화 (kernel=stride=patch_size).
            ref_seq     = img2seq(ref_latent, self.patch_size)                                    # (B, N_ref, patch_dim)
            # get_image_ids_4d() — pierrot_modules.py 정의.
            #   t_offset 만 다르게 줘서 RoPE 회전 단계에서 ref/메인 토큰 분리 신호 부여.
            #   t_offset=10/20/30 으로 RoPE freq 차이 → attention 점수가 자연 분리.
            ref_ids     = get_image_ids_4d(
                bs, h, w, patch_size=self.patch_size, device=ref_latent.device,
                t_offset=t_offset,
            )                                                                                     # (B, N_ref, 4)
            ref_seqs.append(ref_seq)
            ref_ids_all.append(ref_ids)

        # 시퀀스 차원으로 concat — torch.cat(dim=1) 은 N 축 결합. (B, N_1+N_2+..., patch_dim).
        full_ref_seq = torch.cat(ref_seqs,    dim=1)                                              # (B, N_ref_total, patch_dim)
        full_ref_ids = torch.cat(ref_ids_all, dim=1)                                              # (B, N_ref_total, 4)
        # self.pe_embedder(EmbedND) — 합쳐진 ids 한 번에 RoPE 회전 행렬로 변환.
        ref_pe       = self.pe_embedder(full_ref_ids)                                             # (B, 1, N_ref_total, head_dim/2, 2, 2)
        return full_ref_seq, ref_pe

    def forward(
        self,
        image_latent:          Tensor,
        timestep:              Tensor,
        prompt_embeds:         Tensor,
        prompt_attention_mask: Tensor | None = None,
        ref_latents:           list[Tensor] | None = None,
        ref_t_offsets:         list[int]    | None = None,
        tread_step:            int = 0,
    ) -> Tensor:
        """PIERROT 의 최상위 forward.   다중 참조 이미지 입력 지원 (선택).

        Args:
            image_latent:          (B, C, H, W).   생성 대상 (노이즈 섞인 latent).
            timestep:              (B,).   연속값 timestep.
            prompt_embeds:         (B, L_txt, ctx_dim).   텍스트 인코더 출력.
            prompt_attention_mask: None 또는 (B, L_txt).   텍스트 패딩 마스크.
            ref_latents:           None / [] / list of (B, C, H_ref_i, W_ref_i).
            ref_t_offsets:         None 또는 list[int].   각 ref 의 t 좌표.
            tread_step:            TREAD 라우팅 시 결정론적 토큰 샘플링 학습 step.

        Returns:
            (B, C, H, W).   디노이저 출력 (Flow Matching velocity).
        """
        model_timestep    = timestep

        # 1) 표준 전처리 — self.process_inputs() 위 정의.
        #    내부: txt_in(text 사영) + img2seq(패치화) + pe_embedder(메인 이미지 RoPE).
        img_seq, txt, pe = self.process_inputs(image_latent, prompt_embeds)
        # img_seq : (B, N_gen, patch_dim)            ← img2seq 결과 (img_in 은 forward_transformers 안에서)
        # txt     : (B, L_txt, hidden)               ← txt_in 통과 완료
        # pe      : (B, 1, N_gen, head_dim/2, 2, 2)  ← 메인 이미지 RoPE 회전행렬

        # 2) 참조 이미지 처리 — 메인 이미지 시퀀스에 합치지 않고 ref_kv_proj 로 별도 처리.
        #    리스트가 비었거나 None 이면 ref-less T2I 동작 (block 의 ref 분기 진입 X).
        refs    = None
        pe_refs = None
        if ref_latents:
            # self._encode_refs_pre_in() — 위 정의.   각 ref 를 다른 t_offset 으로 patch + 4D ids.
            refs, pe_refs = self._encode_refs_pre_in(ref_latents, ref_t_offsets)              # (B, N_ref, patch_dim), (B, 1, N_ref, D/2, 2, 2)

        # 3) 트랜스포머 스택 + final_layer
        # self.forward_transformers — 위 메서드.   img_in + 모든 block forward + final_layer 까지.
        #   TREAD wrapper 가 모델을 감싸면 이 메서드가 후킹 지점이 됨 (block_kwargs 로 tread_step 전달).
        img_seq = self.forward_transformers(
            img_seq, txt, model_timestep,
            pe=pe, attention_mask=prompt_attention_mask,
            refs=refs, pe_refs=pe_refs,
            tread_step=tread_step,
        )                                                                                     # (B, N_gen, p*p*out_channels)

        # 4) unpatchify → 원본 latent 격자 복원.
        # seq2img() — 위 정의.   fold() 로 (B, N, p²·C) → (B, C, H, W).   image_latent.shape 로 격자 복원.
        return seq2img(img_seq, self.patch_size, image_latent.shape)                          # (B, C, H, W)


if __name__ == "__main__":
    DEVICE        = torch.device("cpu")
    DTYPE         = torch.bfloat16
    TORCH_COMPILE = False

    BS                   = 2
    LATENT_C             = 16
    FEATURE_H, FEATURE_W = 1024 // 8, 1024 // 8
    PROMPT_L             = 120

    pierrot_small_config = {
        "in_channels":     LATENT_C,
        "patch_size":      2,
        "context_in_dim":  2304,
        "hidden_size":     1792,
        "mlp_ratio":       3.5,
        "num_heads":       28,                      # head_dim = 1792 / 28 = 64
        "depth":           16,
        "axes_dim":        [16, 16, 16, 16],        # 4D — sum == head_dim 64
        "theta":           2000,
        "time_factor":     1000.0,
        "time_max_period": 10000,
    }

    denoiser = PIERROT(pierrot_small_config).to(DEVICE, DTYPE)
    print(f"Total parameters: {sum(p.numel() for p in denoiser.parameters()) / 1e9: .3f}B")

    if TORCH_COMPILE:
        denoiser = torch.compile(denoiser)

    # 1024x1024 정방형
    out = denoiser(
        image_latent          = torch.randn(BS, LATENT_C, FEATURE_H, FEATURE_W, device=DEVICE, dtype=DTYPE),
        timestep              = torch.zeros(BS, device=DEVICE, dtype=DTYPE),
        prompt_embeds         = torch.zeros(BS, PROMPT_L, 2304, device=DEVICE, dtype=DTYPE),
        prompt_attention_mask = torch.ones(BS, PROMPT_L, device=DEVICE, dtype=DTYPE),
    )
    out.sum().backward()
    print(f"[1024x1024] OK  output shape = {tuple(out.shape)}")

    # 1248x832 비정방형 (aspect ratio 확장성)
    FEATURE_H, FEATURE_W = 1248 // 8, 832 // 8
    out = denoiser(
        image_latent          = torch.randn(BS, LATENT_C, FEATURE_H, FEATURE_W, device=DEVICE, dtype=DTYPE),
        timestep              = torch.zeros(BS, device=DEVICE, dtype=DTYPE),
        prompt_embeds         = torch.zeros(BS, PROMPT_L, 2304, device=DEVICE, dtype=DTYPE),
        prompt_attention_mask = torch.ones(BS, PROMPT_L, device=DEVICE, dtype=DTYPE),
    )
    out.sum().backward()
    print(f"[1248x832 ] OK  output shape = {tuple(out.shape)}")
