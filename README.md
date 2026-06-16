<p align="center">
  <img src="docs/pierrot-banner.png" width="100%" alt="PIERROT banner"/>
</p>

<h1 align="center">🎭 PIERROT</h1>

<p align="center">
  <b>1인 실험 이미지 생성 모델  프로젝트</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/model-0.857B-success.svg" alt="params"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
</p>

---

## 💡 소개

**PIERROT** 는 1인 이미지 생성 연구·개발 프로젝트입니다. 거대 자원 없이도 작은 모델로 어디까지 갈 수 있는지 실험하는 개인 테스트베드이며, 다음 네 가지를 설계 철학으로 둡니다.

1. **차용과 재현 우선 (MimiC)** — 새로운 알고리즘을 직접 시도하기도 하지만, 기본 방향은 기존 연구들의 검증된 장점을 골라 **재현·결합**하는 것입니다.
2. **저비용 소형 모델 지향** — 학습 방법 · 모델 구조 · 수렴 전략 모든 단계에서 **메모리를 최소화**합니다. 적은 자원으로 가능성을 빠르게 검증하는 실험 장치입니다.
3. **상업 모델에 근접하는 품질** — 가능성이 확인되면, 작은 모델이더라도 **기존 상업용 모델에 근접한 품질**을 목표로 점진적으로 키워 나갑니다.
4. **개인적 호기심의 실험장** — 무엇보다, 만드는 사람의 호기심을 푸는 공간입니다.


## ✨ 특징

- **순수 PyTorch** — 학습 의존성 0. `torch` + `diffusers`(VAE) + `transformers`(text encoder) 만 있으면 동작.
- **자기 완결적(self-contained)** — 모델 · 스케줄러 · 파이프라인 · 토크나이저가 한 패키지 안에 모두 포함.
- **4D RoPE+Hybrid 트랜스포머** — (t, h, w, l) 좌표계로 메인 이미지 + 다중 참조 + 어순을 한 번에 인코딩.
- **Flow Matching 학습/추론** — `x_prediction` / `velocity` 두 모드, 해상도 적응형 `snr_shift` 스케줄.
- **학습 모델과 bit-exact** — 학습에 쓰인 가중치를 그대로 strict 로드, 출력이 학습 그래프와 비트 단위로 동일.

## 🧠 주요 알고리즘

| 알고리즘 | 핵심 역할 | 비고 / 영감 |
|---|---|---|
| **4D RoPE** (t, h, w, l) | 메인 이미지 · 다중 참조 · 텍스트 어순을 한 좌표계로 회전 위치 인코딩 | FLUX.2 RoPE 규약 확장 |
| **Hybrid 블록** | 앞쪽 N개 = 양방향 `PIERROTDualBlock`(MMDiT식, 텍스트도 Q 생성), 나머지 = 비대칭 `PIERROTBlock`(텍스트 KV-only) | MMDiT · PRX |
| **Flow Matching** | 학습/추론 목적함수. `x_prediction`(x₀ 직접 예측 후 velocity 환산) / `velocity` 두 모드 | Rectified Flow |
| **snr_shift 스케줄** | 이미지 토큰 수 기반 μ 로 timestep 을 해상도 적응형 재분배 (어려운 t≈1 구간에 step 집중) | SANA / FLUX |
| **GQA** (`n_kv_heads=4`) | K/V head 공유로 attention 메모리·연산 절감 | LLaMA-2 |
| **AdaLN-Zero** (4-param) | timestep 조건부 scale/gate 변조. shift 제거로 modulation 파라미터 절감 | DiT |
| **Sandwich-Norm + tanh-gate + RMSNorm** | 깊은 모델 bf16 잔차 진폭 안정화 | Z-Image |
| **CFG** (classifier-free guidance) | `guidance_scale` 로 프롬프트 충실도 제어 | — |
| **chi_prompt** (옵션) | text encoder 입력 prefix 로 프롬프트 의미 확장 | SANA (Complex Human Instruct) |

## 📦 구조

```
PIERROT_INFER/
├── sample.py              # CLI 진입점  (python -m PIERROT_INFER.sample)
├── infer.sh               # 다중 프롬프트 일괄 추론 wrapper
├── docs/                  # 배너 · 샘플 이미지
├── results/               # 추론 출력 폴더
├── model/                 # PIERROT 트랜스포머 본체
│   ├── pierrot.py
│   ├── pierrot_modules.py     # 4D RoPE · 비대칭/양방향 블록 · AdaLN
│   └── configs.py             # CONFIG_PRESETS (tiny / 0.8b / 1.6b)
├── scheduler/             # 추론 전용 schedule + step
│   ├── timestep_schedule.py   # linear / linear_shift / snr_shift
│   └── flow_matching.py       # euler_step · x_prediction_to_velocity
└── pipeline/                  # 텍스트→이미지 추론 + 프롬프트 전처리
    ├── sampling.py            # denoise / denoise_cfg
    ├── PIERROTPipeline.py     # 사용자 친화 wrapper
    ├── text_encoding.py       # quote-aware 토크나이저
    └── constants.py           # chi_prompt 기본 템플릿
```

## 🚀 빠른 시작

```bash
# 1) 의존성 설치
pip install torch diffusers transformers safetensors einops pillow

# 2) PIERROT_INFER 를 import 가능한 위치(상위 디렉토리)에 둔다
#    예) <study_root>/PIERROT_INFER  →  cd <study_root>
```

> 텍스트 인코더(`Qwen/Qwen3-4B`)와 VAE(`black-forest-labs/FLUX.2-small-decoder`)는 첫 실행 시 Hugging Face 에서 자동 다운로드됩니다.

## 🎨 추론

### CLI

```bash
cd <study_root>          # PIERROT_INFER 의 상위 디렉토리

CUDA_VISIBLE_DEVICES=0 python -m PIERROT_INFER.sample \
    --ckpt   checkpoints/0.8b_base/model.pt \
    --prompt "a red apple on a wooden table" \
    --output PIERROT_INFER/results/apple.png \
    --steps 28 --seed 42
```

다중 프롬프트 일괄 추론(출력은 `PIERROT_INFER/results/`):

```bash
# 경로는 스크립트 위치 기반 자동 도출. 필요 시 env 로 override:
#   PIERROT_ROOT / CKPT_PATH / OUT_DIR / PYTHON_BIN
CKPT_STEP=02370000 GPU=0 bash PIERROT_INFER/infer.sh
```

### Python API

```python
import torch
from PIERROT_INFER.model import PIERROT
from PIERROT_INFER.model.configs import CONFIG_PRESETS
from PIERROT_INFER.pipeline import PIERROTPipeline

# 모델 빌드 + 체크포인트 로드는 sample.py 의 build_model() 참고
model = PIERROT(CONFIG_PRESETS["0.8b"]).eval()
# ... vae / text_encoder / tokenizer 준비 후 ...
pipe  = PIERROTPipeline(model=model, text_encoder=enc, tokenizer=tok, vae=vae)
image = pipe(prompt="a red apple on a wooden table",
             num_inference_steps=28, guidance_scale=4.0)["images"][0]
```

전체 옵션은 `python -m PIERROT_INFER.sample --help` 를 참고하세요.

## 🔧 알림

- 이 패키지는 현재 공개하지 않는 PIERROT_LAB에서 학습한 추론 코드입니다.
- 현재 학습셋은 공개할 계획을 가지고 있지 않습니다. (향후 고민)
- 1인 프로젝트라 비용에 대한 문제가 존재하여 진행속도가 느릴수 있습니다.
- 참여·조언·채용등의 문의는 언제나 환영입니다.
- 관련 GPU등의 지원은 언제나 기다리고 있습니다. :) 

## 📚 학습 데이터셋

- Pierrot은 다음과 같은 공개 데이터셋을 이용하여 학습에 적용했습니다. (Thanks~)
- 모든 데이터셋을 그대로 사용한게 아니라, 일부는 part / 정제 subset 만 사용합니다.

| 데이터셋 | 공개 출처 | 규모(근사) | 역할 |
|---|---|---:|---|
| **FLUX-Reason-6M** | [LucasFang/FLUX-Reason-6M](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) | 5.89M | reasoning-rich 합성 caption prior |
| **flux_generated** | [lehduong/flux_generated](https://huggingface.co/datasets/lehduong/flux_generated) | 1.75M | 일반 합성 다양성 |
| **CC12M** (WebDataset) | [pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds) | ~3.66M | 웹 캡션 다양성 |
| **DiffusionDB** | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) | ~1.0M | diffusion 프롬프트 다양성 |
| **MONET** (Megalith-10M / COYO) | [jasperai/monet](https://huggingface.co/datasets/jasperai/monet) | ~3.9M | 사진 미학 prior (latent-only) |
| **AnyWord-3M** (LAION) | [stzhao/AnyWord-3M](https://huggingface.co/datasets/stzhao/AnyWord-3M) | ~2.36M | scene-text / OCR prior |
| **HumanCaption-10M** | [OpenFace-CQUPT/HumanCaption-10M](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-10M) | ~0.75M | human-scene prior |
| **DeepFashion-MultiModal** | [yumingj/DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) | 12.7K | full-body fashion prior |



## 🗺️ 로드맵 (To-Do)

**완료**

- [x] 0.8B base 모델 학습 — 1024² multi-aspect
- [x] Flow Matching (`x_prediction`) + 4D RoPE Hybrid 트랜스포머
- [x] 공개 데이터셋 기반 base 사전학습
- [x] 추론 전용 패키지 분리 — 학습 의존성 0
- [x] CLI / Python API 추론 파이프라인
- [x] 다중 참조(multi-reference) 입력 지원 -> 코드만 완료

**진행 / 예정**

- [ ] Post-Training : SFT 
- [ ] Post-Training : DPO 같은 알고리즘 개발 모델 
- [ ] 고해상도 finetune (1280 / 1536)
- [ ] depth growth 스케일업 (→ 2.2B)
- [ ] 체크포인트 공개 (Hugging Face)
- [ ] few-step 모델 개발바 (distillation 버전)
- [ ] 정량 벤치마크 (GenEval / DPG-Bench 등)
- [ ] Edit 모델 개발 
- [ ] Next (새로운 알고리즘) 모델 개발

## 🤗 출처 및 영감

PIERROT 는 기존 연구들의 좋은 점들을 재현·결합하여 만들어집니다.

📚 **참고한 논문 전체 리스트 → [Pierrot-vision/Reading-Papers](https://github.com/Pierrot-vision/Reading-Papers)**
