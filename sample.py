#!/usr/bin/env python3
"""PIERROT standalone inference CLI — 학습된 체크포인트로 한 장 이미지 생성.

흐름:
    1. argparse 로 ckpt / prompt / steps / cfg / size / seed / output 받음
    2. PIERROT 모델 빌드 (CONFIG_PRESETS) + state_dict 로드
    3. VAE / text_encoder / tokenizer 빌드
    4. PIERROTPipeline 인스턴스화 → 호출 → PIL 저장

사용 예 (24h 레시피 default 그대로):
    python -m PIERROT_INFER.sample \\
        --ckpt   /path/to/checkpoints/pierrot_small_24h_v1/final/model.pt \\
        --prompt "a digital painting of a rusty vintage tram on a sandy beach"
        # --output 미지정 시 PIERROT/results/pierrot.png

기본 저장 경로:
    --output default = PIERROT/results/pierrot.png (cwd 무관 자동 검출).
    num-images > 1 이면 _0/_1/... 자동 부여.   parent dir 은 자동 mkdir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# scripts/ 안에서 PIERROT 모듈 import 위해 STUDY 루트를 sys.path 추가
_HERE          = Path(__file__).resolve()
_ROOT          = _HERE.parents[1]                                                   # .../STUDY/
_PIERROT_ROOT  = _HERE.parents[0]                                                   # .../STUDY/PIERROT_INFER/
DEFAULT_OUTPUT = str(_PIERROT_ROOT / "results" / "pierrot.png")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .pipeline import PIERROTPipeline                                              # noqa: E402
from .model import PIERROT                                                          # noqa: E402
from .model.configs import CONFIG_PRESETS                                           # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIERROT inference CLI")

    # 모델
    p.add_argument("--ckpt",        type=str, required=True,
                   help="학습된 PIERROT 모델 .pt 경로 (state_dict) 또는 디렉토리")
    p.add_argument("--config-size", type=str, default="0.8b", choices=list(CONFIG_PRESETS.keys()),
                   help="CONFIG_PRESETS 키 (학습 시와 동일해야 함)")

    # Z-Image 옵션 (24h 레시피 default)
    p.add_argument("--sandwich-norm",      action=argparse.BooleanOptionalAction, default=True,
                   help="Z1: post-norm RMSNorm")
    p.add_argument("--use-tanh-gate",      action=argparse.BooleanOptionalAction, default=True,
                   help="Z2: gate ±1 클램프")
    p.add_argument("--adaln-4param",       action=argparse.BooleanOptionalAction, default=True,
                   help="Z4: shift 제거 modulation")
    p.add_argument("--use-rmsnorm",        action=argparse.BooleanOptionalAction, default=True,
                   help="Z5: LayerNorm→RMSNorm")
    p.add_argument("--axes-max-len",       type=int, nargs=4, default=None,
                   metavar=("T", "H", "W", "L"),
                   help="Z3: 4D RoPE freqs cache (t, h, w, l).   default=None → preset 값 유지")
    p.add_argument("--adaln-embed-dim",    type=int, default=256,
                   help="Z6: modulation 입력 dim cap")
    p.add_argument("--n-kv-heads",         type=int, default=4,
                   help="Z7: GQA K/V head 수")
    # 텍스트 / VAE
    p.add_argument("--text-encoder-id",        type=str, default="Qwen/Qwen3-4B",
                   help="HuggingFace text encoder (학습 precompute 와 동일 모델)")
    p.add_argument("--text-encoder-layers",    type=int, nargs="+", default=[9, 18, 27],
                   help="hidden_states layer 인덱스 (학습 시와 동일)")
    p.add_argument("--vae-id",                 type=str, default="black-forest-labs/FLUX.2-small-decoder",
                   help="HuggingFace VAE")
    p.add_argument("--vae-subfolder",          type=str, default="",
                   help="VAE 가 subfolder 안에 있으면 명시")
    p.add_argument("--use-sana-chi-prompt",   action="store_true",
                   help="Sana 표준 chi_prompt 자동 사용 (training/args.py 의 _CHI_PROMPT_SANA_DEFAULT)")
    p.add_argument("--chi-prompt-file",       type=str, default=None,
                   help="chi_prompt 파일 경로 (각 줄 = chi_prompt 한 줄)")
    p.add_argument("--max-prompt-tokens",      type=int, default=512,
                   help="text encoder 토큰 max_length")
    p.add_argument("--clean-prompt",           action="store_true", default=False,
                   help="deepfloyd-IF 식 prompt 정규화 적용")

    # 추론 옵션
    p.add_argument("--prompt",          type=str, required=True)
    p.add_argument("--negative",        type=str, default="")
    p.add_argument("--steps",           type=int, default=28)
    p.add_argument("--cfg",             type=float, default=4.0)
    p.add_argument("--height",          type=int, default=1024)
    p.add_argument("--width",           type=int, default=1024)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--num-images",      type=int, default=1)
    p.add_argument("--no-binning",      action="store_true",
                   help="resolution binning 비활성")
    p.add_argument("--prediction-mode", type=str, default="x_prediction",
                   choices=("velocity", "x_prediction"))
    p.add_argument("--schedule-method", type=str, default="snr_shift",
                   choices=("linear", "linear_shift", "snr_shift"))
    p.add_argument("--schedule-shift",  type=float, default=1.0,
                   help="schedule-method='linear_shift' 일 때만 사용")
    p.add_argument("--allow-missing-keys", action="store_true", default=False,
                   help="ckpt 의 누락/잉여 key 허용 (strict=False).   default 는 strict=True")

    # 환경
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",  type=str, default="bfloat16",
                   choices=("float32", "float16", "bfloat16"))
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help=f"저장 경로.   default = {DEFAULT_OUTPUT}.   num-images>1 이면 _0/_1/... 자동 부여")

    return p.parse_args()


def load_vae(model_id: str, subfolder: str | None, device: str, dtype: torch.dtype):
    """diffusers VAE 로드 — FLUX.2 → AutoencoderKLFlux2 (구 버전 fallback: AutoencoderKL)."""
    from diffusers import AutoencoderKL
    try:
        from diffusers import AutoencoderKLFlux2 as VAEClass                        # type: ignore
    except ImportError:
        VAEClass = AutoencoderKL                                                    # 구 버전 fallback

    vae = VAEClass.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
    vae.to(device).eval().requires_grad_(False)
    return vae


def load_text_tower(model_id: str, device: str, dtype: torch.dtype):
    """transformers text encoder 로드 (output_hidden_states=True)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    txt = AutoModel.from_pretrained(
        model_id,
        torch_dtype          = dtype,
        trust_remote_code    = True,
        output_hidden_states = True,
    )
    txt.to(device).eval().requires_grad_(False)
    return tok, txt


def build_model(args: argparse.Namespace) -> PIERROT:
    """CONFIG_PRESETS + Z-Image 옵션 → PIERROT 인스턴스 + state_dict 로드."""
    config = CONFIG_PRESETS[args.config_size].copy()

    # Z-Image 옵션
    config["sandwich_norm"] = args.sandwich_norm
    config["use_tanh_gate"] = args.use_tanh_gate
    config["adaln_4param"]  = args.adaln_4param
    config["use_rmsnorm"]   = args.use_rmsnorm

    if args.axes_max_len is not None:
        config["axes_max_len"] = args.axes_max_len
    if args.adaln_embed_dim is not None:
        config["adaln_embed_dim"] = args.adaln_embed_dim
    if args.n_kv_heads is not None:
        config["n_kv_heads"] = args.n_kv_heads

    # ckpt 경로 자동 인식 — 디렉토리면 안에서 model 파일 후보 자동 탐색
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        for cand in ("model_only.pt", "ema_only.pt", "model.safetensors", "pytorch_model.bin"):
            if (ckpt_path / cand).exists():
                ckpt_path = ckpt_path / cand
                print(f"[INFO] ckpt 디렉토리 감지 — 모델 파일로 {ckpt_path.name} 사용")
                break
        else:
            raise FileNotFoundError(
                f"{args.ckpt} 안에 model.safetensors / pytorch_model.bin 둘 다 없음."
            )

    # state_dict 로드 — safetensors 또는 .pt/.bin
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(ckpt_path))                                           # safetensors → dict[str, Tensor]
    else:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # wrapper key 자동 unwrap — (a) 직접, (b) {"model": ...}, (c) {"ema_model": ...}
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "ema_model" in state and isinstance(state["ema_model"], dict):
            state = state["ema_model"]
            print("[INFO] EMA 체크포인트 감지 — ema_model state_dict 사용")

    model = PIERROT(config)

    # strict 로드 정책 — default True, --allow-missing-keys 로 완화
    strict = not args.allow_missing_keys
    if strict:
        model.load_state_dict(state, strict=True)
        print("[INFO] state_dict 엄격 로드 OK (strict=True)")
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] state_dict 누락 키 {len(missing)} 개 (예: {missing[:3]})")
        if unexpected:
            print(f"[WARN] state_dict 잉여 키 {len(unexpected)} 개 (예: {unexpected[:3]})")
        print("[WARN] --allow-missing-keys 활성화 — 모델 일부 weight random init 가능 (디버깅 모드)")
    return model


def main() -> None:
    args  = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # 1. 모델
    print(f"[INFO] PIERROT 모델 빌드: config={args.config_size}, ckpt={args.ckpt}")
    model    = build_model(args).to(args.device, dtype=dtype).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] params = {n_params/1e6:.2f}M ({n_params/1e9:.3f}B)")

    # 2. 외부 모듈
    vae_subfolder = args.vae_subfolder or None
    print(f"[INFO] VAE 로드: {args.vae_id} (subfolder={vae_subfolder})")
    vae = load_vae(args.vae_id, vae_subfolder, args.device, dtype)

    print(f"[INFO] text encoder 로드: {args.text_encoder_id}")
    tokenizer, text_encoder = load_text_tower(args.text_encoder_id, args.device, dtype)

    # 3. Pipeline
    pipe = PIERROTPipeline(
        model               = model,
        text_encoder        = text_encoder,
        tokenizer           = tokenizer,
        vae                 = vae,
        default_sample_size = args.height,
        prediction_mode     = args.prediction_mode,
        schedule_method     = args.schedule_method,
        schedule_shift      = args.schedule_shift,
        text_encoder_layers = tuple(args.text_encoder_layers) if args.text_encoder_layers else None,
        clean_prompt        = args.clean_prompt,
        max_prompt_tokens   = args.max_prompt_tokens,
    )

    # 4. 추론
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # chi_prompt 결정 — file 우선, 그 다음 Sana default, 둘 다 없으면 None
    chi_prompt: list[str] | None = None
    if args.chi_prompt_file is not None:
        with open(args.chi_prompt_file, encoding="utf-8") as f:
            chi_prompt = [ln.rstrip("\n") for ln in f if ln.strip()]
        print(f"[INFO] chi_prompt 로드 (파일): {args.chi_prompt_file}  ({len(chi_prompt)} lines)")
    elif args.use_sana_chi_prompt:
        from .pipeline.constants import CHI_PROMPT_SANA_DEFAULT as _CHI_PROMPT_SANA_DEFAULT
        chi_prompt = list(_CHI_PROMPT_SANA_DEFAULT)
        print(f"[INFO] chi_prompt 로드 (Sana default): {len(chi_prompt)} lines")

    print(f"[INFO] 생성: prompt={args.prompt!r}, steps={args.steps}, cfg={args.cfg}, "
          f"size={args.height}x{args.width}, seed={args.seed}, "
          f"chi_prompt={'ON ('+str(len(chi_prompt))+' lines)' if chi_prompt else 'OFF'}")

    out = pipe(
        prompt                  = args.prompt,
        negative_prompt         = args.negative,
        height                  = args.height,
        width                   = args.width,
        num_inference_steps     = args.steps,
        guidance_scale          = args.cfg,
        num_images_per_prompt   = args.num_images,
        generator               = generator,
        output_type             = "pil",
        return_dict             = True,
        use_resolution_binning  = not args.no_binning,
        chi_prompt              = chi_prompt,
    )
    images = out["images"]

    # 5. 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(images) == 1:
        images[0].save(out_path)
        print(f"[INFO] 저장 완료: {out_path}")
    else:
        stem, suffix = out_path.stem, out_path.suffix
        for i, im in enumerate(images):
            p = out_path.with_name(f"{stem}_{i}{suffix}")
            im.save(p)
            print(f"[INFO] 저장 완료: {p}")


if __name__ == "__main__":
    main()
