# Copyright 2025 The Photoroom and The HuggingFace Teams.  All rights reserved.
# Apache License, Version 2.0 — http://www.apache.org/licenses/LICENSE-2.0
"""PIERROT Inference Pipeline — 사용자 친화 텍스트→이미지 진입점.

설계 원칙:
    1. diffusers 식 외피를 가능한 그대로 유지 (사용자 API 친숙성).
    2. 내부 denoising loop 는 PIERROT 의 `denoise_cfg()` 를 1 줄 호출로 위임.
    3. PIERROT 24h 레시피 (`x_prediction` + `snr_shift`) 를 default 로.
    4. VAE / text_encoder / tokenizer 는 외부에서 주입 받음.
    5. PixArtImageProcessor 의존 제거 → torch.nn.functional.interpolate 로 단순 resize.

핵심 차이:
    - timestep 정규화      : `t / 1000` 식 대신 PIERROT model 은 `t ∈ [0,1]` 직접 받음
                              (`time_factor=1000` 으로 모델 내부 scale).
    - scheduler             : `FlowMatchEulerDiscreteScheduler` 대신
                              `get_schedule()` + `euler_step()` (float).
    - transformer 시그니처  : `encoder_hidden_states=` → `prompt_embeds=`,
                              `attention_mask=` → `prompt_attention_mask=`.
    - prediction_mode       : velocity-only 대신 velocity / x_prediction 둘 다 지원.
    - resolution binning    : PixArtImageProcessor 대신 자체 단순 reimpl.

API 사용 예:
    pipe = PIERROTPipeline(model=prx, text_encoder=enc, tokenizer=tok, vae=vae)
    out  = pipe(prompt="a cat on a beach", num_inference_steps=28, guidance_scale=4.0)
    img  = out["images"][0]    # PIL.Image (output_type="pil" default)
"""
from __future__ import annotations

import html
import re
import urllib.parse as ul
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .text_encoding import tokenize_mixed_quote_char
from .sampling import denoise, denoise_cfg
from ..scheduler.flow_matching import PredictionMode
from ..scheduler.timestep_schedule import ScheduleMethod


# ftfy 는 optional — 없으면 clean_text() 의 ftfy 단계만 skip.
try:
    import ftfy  # type: ignore
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False


# 24h 레시피 baseline — flux_generated 1024×1024 학습 분포.
DEFAULT_RESOLUTION = 1024


#  Resolution binning 상수 — 학습 분포 안에 inference 해상도를 스냅

ASPECT_RATIO_256_BIN = {
    "0.46": [160, 352], "0.6":  [192, 320], "0.78": [224, 288], "1.0":  [256, 256],
    "1.29": [288, 224], "1.67": [320, 192], "2.2":  [352, 160],
}

ASPECT_RATIO_512_BIN = {
    "0.5":  [352, 704], "0.57": [384, 672], "0.6":  [384, 640], "0.68": [416, 608],
    "0.78": [448, 576], "0.88": [480, 544], "1.0":  [512, 512], "1.13": [544, 480],
    "1.29": [576, 448], "1.46": [608, 416], "1.67": [640, 384], "1.75": [672, 384],
    "2.0":  [704, 352],
}

ASPECT_RATIO_1024_BIN = {
    "0.49": [704, 1440],  "0.52": [736, 1408],  "0.53": [736, 1376],  "0.57": [768, 1344],
    "0.59": [768, 1312],  "0.62": [800, 1280],  "0.67": [832, 1248],  "0.68": [832, 1216],
    "0.78": [896, 1152],  "0.83": [928, 1120],  "0.94": [992, 1056],  "1.0":  [1024, 1024],
    "1.06": [1056, 992],  "1.13": [1088, 960],  "1.21": [1120, 928],  "1.29": [1152, 896],
    "1.37": [1184, 864],  "1.46": [1216, 832],  "1.5":  [1248, 832],  "1.71": [1312, 768],
    "1.75": [1344, 768],  "1.87": [1376, 736],  "1.91": [1408, 736],  "2.05": [1440, 704],
}

ASPECT_RATIO_BINS = {
    256:  ASPECT_RATIO_256_BIN,
    512:  ASPECT_RATIO_512_BIN,
    1024: ASPECT_RATIO_1024_BIN,
}


def classify_height_width_bin(
    height: int,
    width:  int,
    ratios: dict[str, list[int]],
) -> tuple[int, int]:
    """요청 (h, w) 를 가장 가까운 aspect-ratio bucket 의 (H, W) 로 스냅.

    학습 분포 외부의 임의 비율 요청을 학습된 bucket 으로 정렬.
    반환된 (H, W) 로 inference 후, 호출부에서 원본 (orig_h, orig_w) 로 resize 하면
    사용자 요청 그대로의 결과 픽셀.

    Args:
        height : int                요청 높이
        width  : int                요청 너비
        ratios : dict[str, [H, W]]  ASPECT_RATIO_*_BIN 중 하나

    Returns:
        (int H, int W)   선택된 bucket 의 (H, W)
    """
    aspect    = float(height) / float(width)
    closest_k = min(ratios.keys(), key=lambda r: abs(aspect - float(r)))           # 비율 차 절댓값 최소
    H, W      = ratios[closest_k]                                                  # [H, W] 리스트
    return H, W


#  TextPreprocessor — 캡션 클렌징 (학습 시와 동일 정규화)

class TextPreprocessor:
    """T5 / Gemma / Qwen 토크나이저 입력 전 캡션 정규화.

    핵심 처리:
        - URL / @멘션 / IP 주소 제거
        - 다양한 dash (—, ‒, …) → "-"
        - 다양한 quote → '"' / "'"
        - HTML entity (&amp; / &quot; …) 디코딩
        - CJK / 일본어 영역 제거 (학습 영어 분포)
        - 파일명 / "1024x1024" / shipping/download 광고 → 제거
        - 알파숫자 spam (jc6640, 9k8h2v 등) → 제거

    ftfy 는 optional (미설치면 skip).
    """

    def __init__(self) -> None:
        # bad punctuation: # ® • © ™ & @ · º ½ ¾ ¿ ¡ § ~ ( ) [ ] { } | \ / *
        self.bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" + r"\{"
            + r"\|" + r"\\" + r"\/" + r"\*"
            + r"]{1,}"
        )

    def clean_text(self, text: str) -> str:
        """raw prompt → 학습 시와 같은 정규화를 거친 cleaned prompt.

        Args:
            text : str   원본 prompt (사용자 입력)

        Returns:
            str   cleaned prompt
        """
        text = str(text)
        text = ul.unquote_plus(text)                                                # %20 등 URL 디코딩
        text = text.strip().lower()
        text = re.sub("<person>", "person", text)

        # URL 제거 — http(s)://, www., 도메인.com/co/... 패턴
        text = re.sub(
            r"\b((?:https?|www):(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@))",
            "", text,
        )

        # @멘션 제거 (소셜 미디어 잔여물)
        text = re.sub(r"@[\w\d]+\b", "", text)

        # CJK / 일본어 / 한자 영역 제거 — 학습 분포는 영어
        text = re.sub(r"[㇀-㇯]+", "", text)
        text = re.sub(r"[ㇰ-ㇿ]+", "", text)
        text = re.sub(r"[㈀-㋿]+", "", text)
        text = re.sub(r"[㌀-㏿]+", "", text)
        text = re.sub(r"[㐀-䶿]+", "", text)
        text = re.sub(r"[䷀-䷿]+", "", text)
        text = re.sub(r"[一-鿿]+", "", text)

        # 다양한 dash → "-" 표준화 — em-dash, en-dash, fullwidth-hyphen 등
        text = re.sub(
            r"[-֊־᐀᠆‐-―⸗⸚⸺⸻⹀〜〰゠︱︲﹘﹣－]+",
            "-", text,
        )
        # 다양한 quote → 표준
        text = re.sub(r"[`´«»" "¨]", '"', text)
        text = re.sub(r"['']", "'", text)
        # HTML entity
        text = re.sub(r"&quot;?", "", text)
        text = re.sub(r"&amp", "", text)
        # IP 주소
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)
        # article id 잔여
        text = re.sub(r"\d:\d\d\s+$", "", text)
        # 줄바꿈 잔여
        text = re.sub(r"\\n", " ", text)
        # 해시태그 / 긴 숫자
        text = re.sub(r"#\d{1,3}\b", "", text)
        text = re.sub(r"#\d{5,}\b", "", text)
        text = re.sub(r"\b\d{6,}\b", "", text)
        # 파일명 — *.png/jpg/...
        text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)
        # 중복 punct
        text = re.sub(r"[\"\']{2,}", r'"', text)
        text = re.sub(r"[\.]{2,}", r" ", text)
        text = re.sub(self.bad_punct_regex, r" ", text)
        text = re.sub(r"\s+\.\s+", r" ", text)

        # 다중 dash/underscore — 4개 이상이면 spam 패턴, 공백으로
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, text)) > 3:
            text = re.sub(regex2, " ", text)

        # ftfy 보정 — optional (미설치 시 skip)
        if _HAS_FTFY:
            text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = text.strip()

        # 알파숫자 spam — jc6640 / jc6640vc / 6640vc231 등 판매 코드 패턴
        text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)
        text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)
        text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)

        # 광고성 단어
        text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
        text = re.sub(r"(free\s)?download(\sfree)?", "", text)
        text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
        text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
        text = re.sub(r"\bpage\s+\d+\b", "", text)
        # 더 긴 알파숫자 spam (j2d1a2a 등) / 해상도 표기 (1024x1024)
        text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)
        text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

        # 마무리 — 공백·콜론·따옴표 정리
        text = re.sub(r"\b\s+\:\s+", r": ", text)
        text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)
        text.strip()
        text = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", text)
        text = re.sub(r"^[\'\_,\-\:;]", r"", text)
        text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
        text = re.sub(r"^\.\S+$", "", text)

        return text.strip()


#  PIERROTPipeline — 메인 inference 진입점

class PIERROTPipeline:
    """PIERROT 의 사용자 친화 inference pipeline.

    diffusers 식 외피 (입력 검증·encode_prompt·prepare_latents·resolution binning·VAE
    decode·output_type 분기) + PIERROT `denoise_cfg()` 의 내부 denoising loop 결합.

    구성 요소:
        model        — PIERROT 트랜스포머.   `model(image_latent=, timestep=, prompt_embeds=, prompt_attention_mask=)` 시그니처.
        text_encoder — `last_hidden_state` 반환하는 인코더.   T5GemmaEncoder / Qwen3Encoder 등.
        tokenizer    — `model_max_length` 속성 보유.   T5/Gemma/Qwen Tokenizer.
        vae          — Optional.   VAE 가 None 이면 `output_type='latent'/'pt'` 만 가능.

    특징:
        - DiffusionPipeline 의존 없음 — 단순 객체 보관·`__call__` 노출.
        - `from_pretrained()` 미구현 — 인스턴스 직접 주입 패턴.
        - prediction_mode / schedule_method / schedule_shift 가 24h 레시피에 맞춘 default.
    """

    def __init__(
        self,
        model:               Any,                          # PIERROT 모델
        text_encoder:        Any,                          # T5GemmaEncoder / Qwen 등
        tokenizer:           Any,                          # *Tokenizer*Fast
        vae:                 Any | None = None,            # AutoencoderKL / AutoencoderDC
        default_sample_size: int = DEFAULT_RESOLUTION,
        # prediction / schedule default — PIERROT 학습이 'x_prediction' + 'snr_shift' 정합.
        prediction_mode: PredictionMode = "x_prediction",
        schedule_method: ScheduleMethod = "snr_shift",
        schedule_shift:  float          = 1.0,
        # text encoder hidden 추출 방식 — 학습 시 인코딩 방식과 정합 필수.
        #   None              : last_hidden_state 사용 (T5-Gemma / 단순 인코더 default)
        #   tuple[int, ...]   : hidden_states[i] for i in layers 마지막 dim 으로 concat
        #                       예: (9, 18, 27) → Qwen3-4B 24h 레시피 (context_in_dim=7680)
        text_encoder_layers: tuple[int, ...] | None = None,
        # clean_prompt — default OFF (학습 정합).   학습이 raw caption 그대로 토크나이즈하면 그대로.
        clean_prompt: bool = False,
        # max_prompt_tokens — 학습과 동일 default 512.
        # tokenizer.model_max_length (Qwen3-4B = 40960) 그대로 쓰면 OOM + 학습 분포 불일치.
        max_prompt_tokens: int = 512,
        quote_char_encode: bool = True,
        quote_char_encode_preserve_quoted: bool = True,
    ):
        """파이프라인 인스턴스 초기화.

        Args:
            model               : PIERROT 모델
            text_encoder        : T5/Gemma/Qwen 인코더
            tokenizer           : 토크나이저
            vae                 : VAE (없으면 latent/pt 만 가능)
            default_sample_size : 기본 해상도 (binning bucket key 로도 사용)
            prediction_mode     : "velocity" | "x_prediction"
            schedule_method     : "linear" | "linear_shift" | "snr_shift"
            schedule_shift      : linear_shift 강도
            text_encoder_layers : None 또는 hidden layer 인덱스 tuple
            clean_prompt        : True 면 clean_text() 적용
            max_prompt_tokens   : tokenizer padding max_length (학습과 정합 필수)
            quote_char_encode   : True 면 "..." 내부 literal text 를 문자 단위로 tokenization

        예외:
            ValueError — VAE latent channels 와 model.in_channels mismatch.
        """
        self.model                = model
        self.text_encoder         = text_encoder
        self.tokenizer            = tokenizer
        self.vae                  = vae
        self.default_sample_size  = default_sample_size
        self.prediction_mode      = prediction_mode
        self.schedule_method      = schedule_method
        self.schedule_shift       = schedule_shift
        self.text_encoder_layers  = text_encoder_layers
        self.clean_prompt         = clean_prompt
        self.max_prompt_tokens    = max_prompt_tokens
        self.quote_char_encode    = quote_char_encode
        self.quote_char_encode_preserve_quoted = quote_char_encode_preserve_quoted

        self.text_preprocessor = TextPreprocessor()

        # VAE latent channels vs model in_channels 정합 검증 — 빠른 실패.
        if vae is not None:
            try:
                vae_ch = _get_latent_channels(vae, model=None)              # wrapper/VAE 자체 정합 검증
            except AttributeError:
                vae_ch = None                                                # 속성 없는 외부 wrapper 면 skip
            model_ch = int(getattr(model, "in_channels", -1))
            if vae_ch is not None and model_ch > 0 and vae_ch != model_ch:
                raise ValueError(
                    f"VAE latent_channels ({vae_ch}) != model.in_channels ({model_ch}). "
                    f"학습 시 사용한 VAE 와 동일 모델인지 확인하라."
                )

        # CFG state — 호출별로 갱신.
        self._guidance_scale: float = 1.0

    # properties

    @property
    def vae_scale_factor(self) -> int:
        """VAE 의 spatial 압축률.   이미지 (H, W) → latent (H/scale, W/scale).

        우선순위 (위에서부터):
            1. `vae.spatial_compression_ratio`         DC-AE 표준
            2. `vae.scale_factor`                      wrapper alias
            3. `vae.config.spatial_compression_ratio`  config 안에 정의된 케이스
            4. `vae.config.scale_factor`               일부 모델 (FLUX 변형 등)
            5. `2 ** (len(config.block_out_channels) - 1)`   FLUX/SDXL/AutoencoderKL 식
            6. 8                                       최종 fallback

        주의: `config.scaling_factor` 는 latent magnitude scale 이라 spatial 의미 X.
        """
        if self.vae is None:
            return 8
        # 1. spatial_compression_ratio (속성 또는 config) — DC-AE 표준
        if hasattr(self.vae, "spatial_compression_ratio"):
            return int(self.vae.spatial_compression_ratio)
        # 2. scale_factor (wrapper alias)
        if hasattr(self.vae, "scale_factor"):
            return int(self.vae.scale_factor)
        # 3. config.spatial_compression_ratio
        cfg = getattr(self.vae, "config", None)
        if cfg is not None and hasattr(cfg, "spatial_compression_ratio"):
            return int(cfg.spatial_compression_ratio)
        # 4. config.scale_factor
        if cfg is not None and hasattr(cfg, "scale_factor"):
            return int(cfg.scale_factor)
        # 5. block_out_channels 기반 — FLUX/SDXL/AutoencoderKL 식
        if cfg is not None and hasattr(cfg, "block_out_channels"):
            return 2 ** (len(cfg.block_out_channels) - 1)
        # 6. 최종 fallback
        return 8

    @property
    def do_classifier_free_guidance(self) -> bool:
        """guidance_scale > 1.0 이면 CFG 활성."""
        return self._guidance_scale > 1.0

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def device(self) -> torch.device:
        """모델 파라미터의 device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """모델 파라미터의 dtype."""
        return next(self.model.parameters()).dtype

    # 입력 검증

    def check_inputs(
        self,
        prompt:                 str | list[str] | None,
        height:                 int,
        width:                  int,
        guidance_scale:         float,
        prompt_embeds:          Tensor | None = None,
        negative_prompt_embeds: Tensor | None = None,
    ) -> None:
        """입력 인자 sanity 체크.

        검증 항목:
            - prompt 와 prompt_embeds 동시 제공 / 둘 다 None 금지
            - prompt 타입 (str / list)
            - CFG 시 negative_prompt_embeds 도 함께 (precompute 모드)
            - height / width 가 vae_scale_factor 의 배수
            - guidance_scale >= 1.0
        """
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("`prompt` 와 `prompt_embeds` 중 하나만 제공해야 한다.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("`prompt` 또는 `prompt_embeds` 중 하나는 반드시 필요하다.")

        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` 타입은 str 또는 list 여야 한다 (받음: {type(prompt)}).")

        if prompt_embeds is not None and guidance_scale > 1.0 and negative_prompt_embeds is None:
            raise ValueError(
                "`prompt_embeds` 가 주어지고 guidance_scale > 1.0 이면 "
                "`negative_prompt_embeds` 도 함께 제공해야 한다 (CFG 페어).",
            )

        scale = self.vae_scale_factor
        if height % scale != 0 or width % scale != 0:
            raise ValueError(
                f"height ({height}) 와 width ({width}) 는 vae_scale_factor "
                f"({scale}) 의 배수여야 한다.",
            )

        if guidance_scale < 1.0:
            raise ValueError(f"guidance_scale 은 >= 1.0 이어야 한다 (받음: {guidance_scale}).")

    # 텍스트 인코딩

    def _tokenize_prompts(
        self,
        prompts: list[str],
        device:  torch.device,
        prefix:  str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """prompt 리스트 → (input_ids, attention_mask).

        Args:
            prompts : list[str]      raw prompt.   self.clean_prompt=True 면 clean_text 적용
            device  : torch.device

        Returns:
            input_ids      : (B, L) long          L = self.max_prompt_tokens (default 512)
            attention_mask : (B, L) bool
        """
        # clean_prompt 분기
        prompts_to_tokenize = (
            [self.text_preprocessor.clean_text(t) for t in prompts] if self.clean_prompt
            else list(prompts)
        )

        # max_length = self.max_prompt_tokens (학습과 동일 default 512)
        if self.quote_char_encode:
            tokens = tokenize_mixed_quote_char(
                self.tokenizer,
                prompts_to_tokenize,
                max_length=self.max_prompt_tokens,
                quote_char_enabled=[True] * len(prompts_to_tokenize),
                preserve_quoted=self.quote_char_encode_preserve_quoted,
                padding="max_length",
                return_tensors="pt",
                prefix=prefix,                                                       # chi system prompt 는 quote 처리 제외
            )
        else:
            tokens = self.tokenizer(
                prompts_to_tokenize,
                padding              = "max_length",
                max_length           = self.max_prompt_tokens,
                truncation           = True,
                return_attention_mask= True,
                return_tensors       = "pt",
            )
        return tokens["input_ids"].to(device), tokens["attention_mask"].bool().to(device)

    def _encode_prompt_standard(
        self,
        prompt:                      list[str],
        device:                      torch.device,
        do_classifier_free_guidance: bool,
        negative_prompt:             str = "",
        chi_prompt:                  list[str] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """배치 한 번에 cond + uncond 인코딩.

        CFG 시 [neg_prompts; pos_prompts] 로 concat → 한 번에 인코더 → split.
        하나의 forward pass 로 두 종류의 embedding 동시 획득.

        chi_prompt (SANA Complex Human Instruct):
            None / [] (default) — 기존 동작 정확히 동일.
            list[str]           — chi_prompt 줄을 \\n 으로 join 해 prompt 앞 prepend,
                                   tokenizer max_length 를 chi_prompt 토큰 수만큼 확장,
                                   forward 후 `[BOS] + last (max_prompt_tokens-1)` 로 슬라이싱.
                                   결과 shape (B, max_prompt_tokens, D) — chi_prompt 없을 때와 동일.

        Args:
            prompt                      : list[str]
            device                      : torch.device
            do_classifier_free_guidance : bool
            negative_prompt             : str
            chi_prompt                  : list[str] | None

        Returns:
            (text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask)
            CFG off 면 uncond 두 개는 None.
        """
        batch_size = len(prompt)

        # chi_prompt prepend (학습 패턴과 동일)
        chi_prompt_str: str | None = None
        chi_max_length:  int       = self.max_prompt_tokens                          # default = no extension
        if chi_prompt:
            chi_prompt_str = "\n".join(chi_prompt)
            chi_n_tokens   = len(self.tokenizer.encode(chi_prompt_str))
            chi_max_length = chi_n_tokens + self.max_prompt_tokens - 2               # magic 2: [bos], [_]
            prompt = [chi_prompt_str + p for p in prompt]                            # 모든 cond prompt 에 prepend

        if do_classifier_free_guidance:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            # negative prompt 도 같은 chi_prompt prepend (CFG 두 분포 정합)
            if chi_prompt_str is not None:
                negative_prompt = [chi_prompt_str + np for np in negative_prompt]
            prompts_to_encode = negative_prompt + prompt                            # [neg₀..neg_B, pos₀..pos_B]
        else:
            prompts_to_encode = prompt

        # chi_prompt 활성 시 max_length 임시 확장 후 복원.
        _saved_max = self.max_prompt_tokens
        if chi_prompt_str is not None:
            self.max_prompt_tokens = chi_max_length
        try:
            input_ids, attention_mask = self._tokenize_prompts(prompts_to_encode, device, prefix=chi_prompt_str)
        finally:
            self.max_prompt_tokens = _saved_max                                     # 항상 복원 (예외 시에도)

        with torch.no_grad():
            # output_hidden_states=True 가 hidden_states tuple 반환의 전제 — 두 모드 모두에 필요
            out = self.text_encoder(
                input_ids            = input_ids,
                attention_mask       = attention_mask,
                output_hidden_states = True,
            )

            # text_encoder_layers 분기 — 학습 시 인코딩 방식과 정합
            #   None  : last_hidden_state 만 — T5-Gemma 기본 동작 (D = hidden_size)
            #   tuple : hidden_states[i] concat — Qwen3-4B 24h 레시피 (D = hidden_size × len(layers))
            if self.text_encoder_layers is None:
                embeddings = out["last_hidden_state"]                               # (2B or B, L, D)
            else:
                hs = out.hidden_states                                              # tuple, len = num_layers + 1
                embeddings = torch.cat(
                    [hs[i] for i in self.text_encoder_layers],                      # 각 (2B or B, L, hidden)
                    dim=-1,
                )                                                                   # (2B or B, L, hidden × len(layers))

        # chi_prompt 활성 시 select_index 슬라이싱 — [BOS] + last (max_prompt_tokens-1).
        # 결과 shape (B, max_prompt_tokens, D) — chi_prompt 없을 때와 동일.
        if chi_prompt_str is not None:
            sel = [0] + list(range(-self.max_prompt_tokens + 1, 0))
            embeddings     = embeddings[:, sel, :]                                  # (2B or B, max_prompt_tokens, D)
            attention_mask = attention_mask[:, sel]                                 # (2B or B, max_prompt_tokens)

        if do_classifier_free_guidance:
            uncond_emb,  text_emb     = embeddings.split(batch_size, dim=0)         # 각 (B, L, D)
            uncond_mask, text_mask    = attention_mask.split(batch_size, dim=0)     # 각 (B, L)
            return text_emb, text_mask, uncond_emb, uncond_mask
        return embeddings, attention_mask, None, None

    def encode_prompt(
        self,
        prompt:                         str | list[str] | None,
        device:                         torch.device,
        do_classifier_free_guidance:    bool,
        negative_prompt:                str = "",
        num_images_per_prompt:          int = 1,
        prompt_embeds:                  Tensor | None = None,
        negative_prompt_embeds:         Tensor | None = None,
        prompt_attention_mask:          Tensor | None = None,
        negative_prompt_attention_mask: Tensor | None = None,
        chi_prompt:                     list[str] | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """encode_prompt 공식 진입점.

        prompt 가 주어지면 인코딩, 아니면 precompute 된 *_embeds 를 그대로 사용.
        num_images_per_prompt > 1 이면 batch 차원 복제.

        Args:
            prompt                          : str | list[str] | None
            device                          : torch.device
            do_classifier_free_guidance     : bool
            negative_prompt                 : str
            num_images_per_prompt           : int
            prompt_embeds                   : (B, L, D) | None    precompute embedding
            negative_prompt_embeds          : (B, L, D) | None
            prompt_attention_mask           : (B, L)    | None
            negative_prompt_attention_mask  : (B, L)    | None
            chi_prompt                      : list[str] | None

        Returns:
            (text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask)
        """
        # 1) 인코딩 (prompt → embedding) 또는 precompute 사용
        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            text_embeddings, cross_attn_mask, uncond_text_embeddings, uncond_cross_attn_mask = (
                self._encode_prompt_standard(prompt, device, do_classifier_free_guidance, negative_prompt,
                                              chi_prompt=chi_prompt)
            )
        else:
            # precompute 경로 — 외부에서 미리 인코딩된 결과 주입
            text_embeddings        = prompt_embeds.to(device)
            cross_attn_mask        = prompt_attention_mask.to(device) if prompt_attention_mask is not None else None
            uncond_text_embeddings = negative_prompt_embeds.to(device)         if negative_prompt_embeds         is not None else None
            uncond_cross_attn_mask = negative_prompt_attention_mask.to(device) if negative_prompt_attention_mask is not None else None

        # 2) num_images_per_prompt > 1 이면 batch 차원 복제
        if num_images_per_prompt > 1:
            text_embeddings = _repeat_along_batch(text_embeddings, num_images_per_prompt)
            if cross_attn_mask is not None:
                cross_attn_mask = _repeat_along_batch(cross_attn_mask, num_images_per_prompt)
            if do_classifier_free_guidance and uncond_text_embeddings is not None:
                uncond_text_embeddings = _repeat_along_batch(uncond_text_embeddings, num_images_per_prompt)
                if uncond_cross_attn_mask is not None:
                    uncond_cross_attn_mask = _repeat_along_batch(uncond_cross_attn_mask, num_images_per_prompt)

        return (
            text_embeddings,
            cross_attn_mask,
            uncond_text_embeddings if do_classifier_free_guidance else None,
            uncond_cross_attn_mask if do_classifier_free_guidance else None,
        )

    # Latent 초기화

    def prepare_latents(
        self,
        batch_size:           int,
        num_channels_latents: int,
        height:               int,
        width:                int,
        dtype:                torch.dtype,
        device:               torch.device,
        generator:            torch.Generator | list[torch.Generator] | None = None,
        latents:              Tensor | None = None,
    ) -> Tensor:
        """초기 노이즈 latent 또는 외부 주입 latent 정규화.

        generator 가 list 면 sample 별로 다른 seed 적용:
            None | Generator   → 일괄 randn
            list[Generator]    → sample 별 randn 후 stack

        Args:
            batch_size, num_channels_latents, height, width : int
            dtype, device                                    : torch.dtype, torch.device
            generator : None | torch.Generator | list[torch.Generator]
            latents   : (B, C, h, w) Tensor | None    외부 주입 시 사용

        Returns:
            (B, C, height/scale, width/scale) Tensor
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        scale  = self.vae_scale_factor
        h_lat  = height // scale
        w_lat  = width  // scale
        shape  = (batch_size, num_channels_latents, h_lat, w_lat)

        # generator 가 list 면 sample 별 다른 seed
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"generator 리스트 길이 ({len(generator)}) 가 batch_size "
                    f"({batch_size}) 와 일치해야 한다.",
                )
            samples = [
                torch.randn(shape[1:], generator=generator[i], device=device, dtype=dtype)  # (C, h, w)
                for i in range(batch_size)
            ]
            return torch.stack(samples, dim=0)                                              # (B, C, h, w)

        return torch.randn(shape, generator=generator, device=device, dtype=dtype)          # (B, C, h, w)

    # 메인 진입점

    @torch.no_grad()
    def __call__(
        self,
        prompt:                         str | list[str] | None = None,
        negative_prompt:                str = "",
        height:                         int | None = None,
        width:                          int | None = None,
        num_inference_steps:            int = 28,
        guidance_scale:                 float = 4.0,
        num_images_per_prompt:          int = 1,
        generator:                      torch.Generator | list[torch.Generator] | None = None,
        latents:                        Tensor | None = None,
        prompt_embeds:                  Tensor | None = None,
        negative_prompt_embeds:         Tensor | None = None,
        prompt_attention_mask:          Tensor | None = None,
        negative_prompt_attention_mask: Tensor | None = None,
        # 다중 참조 이미지 — 사용자가 list[Tensor] (각 (B, C, H_ref, W_ref) latent) 직접 주입.
        # pipeline 은 이미 latent 공간이라 VAE encode 는 호출자 책임.
        # None / [] (default) 면 기존 표준 T2I 동작 그대로.
        ref_latents:                    list[Tensor] | None = None,
        ref_t_offsets:                  list[int]    | None = None,
        output_type:                    str = "pil",
        return_dict:                    bool = True,
        use_resolution_binning:         bool = True,
        callback_on_step_end:           Callable[[int, float, dict[str, Tensor]], None] | None = None,
        # SANA chi_prompt — None / [] (default) 면 기존 동작 정확히 동일.
        chi_prompt:                     list[str] | None = None,
    ) -> dict[str, Any] | tuple:
        """Text → image 생성.   외피만 diffusers 식, 내부 denoising 은 `denoise_cfg()` 호출.

        흐름:
            0. height/width default + resolution binning 적용
            1. check_inputs
            2. encode_prompt → (text_emb, mask, uncond_emb, uncond_mask)
            3. prepare_latents — 초기 노이즈 (B, C_lat, h_lat, w_lat)
            4. denoise_cfg(...) 또는 denoise(...) — 전체 denoising loop
            5. VAE decode (output_type 분기)
            6. resolution binning → 원래 (orig_h, orig_w) 로 resize
            7. output_type 변환 (PIL / np / pt / latent)

        Args:
            prompt                : str | list[str] | None
            negative_prompt       : str
            height / width        : int | None     None 이면 default_sample_size
            num_inference_steps   : int
            guidance_scale        : float          > 1.0 이면 CFG 활성
            num_images_per_prompt : int
            generator             : torch.Generator | list[torch.Generator] | None
            latents               : (B, C, h, w) Tensor | None     외부 주입 latent
            prompt_embeds         : (B, L, D)         | None       precompute 경로
            negative_prompt_embeds: (B, L, D)         | None
            prompt_attention_mask : (B, L)            | None
            negative_prompt_attention_mask : (B, L)   | None
            ref_latents           : list[Tensor]      | None       다중 참조 latent
            ref_t_offsets         : list[int]         | None       ref t-축 오프셋
            output_type           : "pil" | "np" | "pt" | "latent"
            return_dict           : True 면 dict, False 면 tuple
            use_resolution_binning: 학습된 aspect bucket 으로 (H, W) 스냅 후 원본으로 resize-back
            callback_on_step_end  : 현재 미지원 (None 이외면 NotImplementedError)
            chi_prompt            : list[str] | None  SANA Complex Human Instruct

        Returns:
            dict[str, Any] | tuple   `{"images": ...}` 또는 `(images,)`
        """
        # callback_on_step_end 미지원 — fail-fast.
        if callback_on_step_end is not None:
            raise NotImplementedError(
                "PIERROT sampling.denoise_cfg() 가 step 단위 hook 미지원 — "
                "callback_on_step_end 사용 불가."
            )

        # 0. 해상도 default + binning
        height = height or self.default_sample_size
        width  = width  or self.default_sample_size

        orig_height, orig_width = height, width                                     # binning 끝나도 이 값으로 resize-back
        if use_resolution_binning:
            if self.default_sample_size in ASPECT_RATIO_BINS:
                ratios       = ASPECT_RATIO_BINS[self.default_sample_size]
                height, width = classify_height_width_bin(height, width, ratios)
            # default_sample_size 가 256/512/1024 외 값이면 binning skip — 사용자 (h, w) 그대로 사용

        # 1. 입력 검증
        self.check_inputs(prompt, height, width, guidance_scale, prompt_embeds, negative_prompt_embeds)

        # VAE 가 None 이면 픽셀 출력 불가 — latent / pt 만 허용
        if self.vae is None and output_type not in ("latent", "pt"):
            raise ValueError(
                f"VAE 가 없어 output_type='{output_type}' 불가.  "
                "VAE 를 주입하거나 output_type='latent' / 'pt' 로 호출하라.",
            )

        # batch_size 결정
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]                                     # type: ignore[union-attr]

        device = self.device
        dtype  = self.dtype

        self._guidance_scale = guidance_scale                                       # property 가 참조

        # 2. encode_prompt
        text_emb, text_mask, uncond_emb, uncond_mask = self.encode_prompt(
            prompt                          = prompt,
            device                          = device,
            do_classifier_free_guidance     = self.do_classifier_free_guidance,
            negative_prompt                 = negative_prompt,
            num_images_per_prompt           = num_images_per_prompt,
            prompt_embeds                   = prompt_embeds,
            negative_prompt_embeds          = negative_prompt_embeds,
            prompt_attention_mask           = prompt_attention_mask,
            negative_prompt_attention_mask  = negative_prompt_attention_mask,
            chi_prompt                      = chi_prompt,
        )

        # 3. prepare_latents — 초기 노이즈
        if self.vae is not None:
            num_channels_latents = _get_latent_channels(self.vae, model=self.model)  # 일반 path
        else:
            num_channels_latents = self.model.in_channels                           # vae=None fallback
        latents = self.prepare_latents(
            batch_size            = batch_size * num_images_per_prompt,
            num_channels_latents  = num_channels_latents,
            height                = height,
            width                 = width,
            dtype                 = dtype,
            device                = device,
            generator             = generator,
            latents               = latents,
        )                                                                           # (B*N, C_lat, h_lat, w_lat)

        # ref_latents 정규화 — (a) device/dtype 일치, (b) num_images_per_prompt 복제.
        # ref_latents=None 면 변화 0.   이미 (device, dtype) 일치면 .to() 가 no-op.
        if ref_latents is not None:
            ref_latents = [r.to(device=device, dtype=dtype) for r in ref_latents]
            if num_images_per_prompt > 1:
                ref_latents = [_repeat_along_batch(r, num_images_per_prompt) for r in ref_latents]

        # 4. denoising loop — PIERROT sampling 호출
        if self.do_classifier_free_guidance:
            if uncond_emb is None:
                raise RuntimeError(
                    "CFG 활성화 (guidance_scale > 1.0) 인데 uncond embedding 이 없다.  "
                    "encode_prompt 흐름을 확인하라.",
                )
            latents = denoise_cfg(
                model                         = self.model,
                img                           = latents,
                prompt_embeds_cond            = text_emb,
                prompt_embeds_uncond          = uncond_emb,
                num_steps                     = num_inference_steps,
                guidance_scale                = guidance_scale,
                schedule_method               = self.schedule_method,
                schedule_shift                = self.schedule_shift,
                prediction_mode               = self.prediction_mode,
                prompt_attention_mask_cond    = text_mask,
                prompt_attention_mask_uncond  = uncond_mask,
                ref_latents                   = ref_latents,
                ref_t_offsets                 = ref_t_offsets,
            )
        else:
            latents = denoise(
                model                 = self.model,
                img                   = latents,
                prompt_embeds         = text_emb,
                num_steps             = num_inference_steps,
                schedule_method       = self.schedule_method,
                schedule_shift        = self.schedule_shift,
                prediction_mode       = self.prediction_mode,
                prompt_attention_mask = text_mask,
                ref_latents           = ref_latents,
                ref_t_offsets         = ref_t_offsets,
            )

        # 5/6/7. VAE decode + resize-back + output_type 변환
        if output_type == "latent" or (output_type == "pt" and self.vae is None):
            # 픽셀 변환 없이 latent 또는 pt latent 그대로 반환
            image = latents
        else:
            image = _vae_decode(self.vae, latents)                                  # (B, 3, H, W) [0, 1]

            # binning 적용했으면 원본 해상도로 resize
            if use_resolution_binning and (image.shape[-2:] != (orig_height, orig_width)):
                image = F.interpolate(
                    image,
                    size          = (orig_height, orig_width),
                    mode          = "bilinear",
                    align_corners = False,
                )

            # output_type 변환
            image = _postprocess(image, output_type)

        if not return_dict:
            return (image,)
        return {"images": image}


#  내부 헬퍼

def _repeat_along_batch(tensor: Tensor, n: int) -> Tensor:
    """(B, ...) → (B*n, ...) — sample 별로 n 번 복제.

    예:  (2, L, D), n=3 → (6, L, D)   순서 [s0, s0, s0, s1, s1, s1]

    Args:
        tensor : (B, ...)
        n      : int       복제 횟수

    Returns:
        (B*n, ...)
    """
    bs   = tensor.shape[0]
    rest = tensor.shape[1:]
    return tensor.unsqueeze(1).expand(bs, n, *rest).reshape(bs * n, *rest)


#  VAE 호환 헬퍼 — direct diffusers VAE / wrapper 둘 다 인식

def _get_latent_channels(vae: Any, model: Any | None = None) -> int:
    """VAE 의 latent 채널 수.   wrapper / direct VAE 모두 인식.

    우선순위 (위에서부터):
        1. `vae.vae_channels`           wrapper 의 alias property
        2. `vae.latent_channels`        wrapper 의 직접 속성
        3. `vae.config.latent_channels` direct diffusers VAE 표준
        4. `vae.config.in_channels`     일부 변형
        5. `model.in_channels`          model 측 정합 (assert 단계의 fallback)

    Args:
        vae   : Any        VAE 객체 (wrapper / direct)
        model : Any | None 보조 fallback 용

    Returns:
        int   latent 채널 수

    예외:
        AttributeError — 어떤 경로로도 채널 수를 찾지 못한 경우.
    """
    for attr in ("vae_channels", "latent_channels"):
        if hasattr(vae, attr):
            return int(getattr(vae, attr))
    cfg = getattr(vae, "config", None)
    if cfg is not None:
        for attr in ("latent_channels", "in_channels"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
    if model is not None and hasattr(model, "in_channels"):
        return int(model.in_channels)
    raise AttributeError(
        "VAE 객체에서 latent 채널 수를 찾을 수 없음.   "
        "vae.vae_channels / vae.latent_channels / vae.config.latent_channels / "
        "vae.config.in_channels / model.in_channels 중 하나는 정의돼야 한다."
    )


def _vae_decode(vae: Any, latents: Tensor) -> Tensor:
    """VAE decode — direct diffusers VAE / wrapper 자동 분기.

    **출력 범위 보장**: 반환은 항상 [0, 1] 범위 (B, 3, H, W) Tensor.
    호출부 (`_postprocess`) 가 그 가정 위에 동작하므로, 새 분기 추가 시 같은 범위로 변환 필수.

    분기:
        - wrapper (`hasattr(vae, "unscale_latent")`):
            wrapper 가 이미 [0, 1] 범위로 normalize 후 반환 → 그대로 반환.
        - direct diffusers VAE (default):
            VAE 출력은 [-1, 1] 범위 → `(image / 2 + 0.5).clamp(0, 1)` 로 [0, 1] 변환 후 반환.
            unscale 은 우리가 처리: `(latents / scaling_factor) + shift_factor`.

    None 안전 처리:
        scaling_factor / shift_factor 가 config 에 없거나 None 이면 적용 자체 skip
        (학습이 unscale 없이 진행된 케이스 정합).

    Args:
        vae     : Any        VAE 객체
        latents : (B, C, h, w)

    Returns:
        (B, 3, H, W) Tensor   [0, 1] 범위
    """
    if hasattr(vae, "unscale_latent"):
        # wrapper 식 — wrapper 가 이미 [0, 1] 변환 완료, 그대로 반환
        return vae.decode(latents)

    # direct diffusers VAE 식
    cfg = vae.config

    # scaling/shift attr 가 없거나 None 이면 적용 자체 skip (학습 정합).
    scaling = getattr(cfg, "scaling_factor", None)
    shift   = getattr(cfg, "shift_factor",   None)

    latents_dec = latents
    if scaling is not None:
        latents_dec = latents_dec / float(scaling)              # 학습이 × scaling_factor 한 경우만
    if shift is not None:
        latents_dec = latents_dec + float(shift)                # shift_factor 정의된 모델만
    image = vae.decode(latents_dec, return_dict=False)[0]                           # (B, 3, H, W) [-1, 1]

    # 출력 범위 통일 — [-1, 1] → [0, 1].
    return (image / 2.0 + 0.5).clamp(0.0, 1.0)                                      # (B, 3, H, W) [0, 1]


def _postprocess(image: Tensor, output_type: str) -> Any:
    """VAE decode 결과 ([0, 1] 범위 (B, 3, H, W) Tensor) → 사용자 포맷 변환.

    **입력 범위 가정**: `_vae_decode()` 가 항상 [0, 1] 범위로 통일해 반환.
    본 함수에서 추가 denormalize 불필요 — 안전 clamp 만 한 번 더.

    출력 포맷:
        "pt"    : (B, 3, H, W) Tensor [0, 1]
        "np"    : (B, H, W, 3) numpy uint8 [0, 255]
        "pil"   : list[PIL.Image]

    Args:
        image       : (B, 3, H, W) Tensor   [0, 1]
        output_type : "pt" | "np" | "pil"

    Returns:
        Tensor | ndarray | list[PIL.Image]
    """
    # 안전 clamp — 부동소수 누적 오차로 약간 벗어날 수 있으므로 한 번 더.
    image = image.clamp(0.0, 1.0)

    if output_type == "pt":
        return image                                                                # (B, 3, H, W)

    # CHW → HWC, [0,1] float → [0,255] uint8 numpy
    image_np = image.permute(0, 2, 3, 1).cpu().float().numpy()                      # (B, H, W, 3)
    image_np = (image_np * 255.0).round().astype("uint8")

    if output_type == "np":
        return image_np                                                             # ndarray (B, H, W, 3)

    if output_type == "pil":
        from PIL import Image                                                       # 지연 import — 의존 최소화
        return [Image.fromarray(arr) for arr in image_np]

    raise ValueError(f"output_type '{output_type}' 미지원.   pt / np / pil / latent 중 하나.")
