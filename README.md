<p align="center">
  <img src="docs/pierrot-banner.png" width="100%" alt="PIERROT banner"/>
</p>

<h1 align="center">🎭 PIERROT</h1>

<p align="center">
  <b>1인 이미지 생성 모델 개발 프로젝트</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/model-0.857B-success.svg" alt="params"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg" alt="license"/>
</p>

<p align="center">
  <b>한국어</b> | <a href="README_en.md">English</a>
</p>

---

## 💡 소개

**PIERROT** 는 1인 **Text-to-Image (T2I)** 생성 연구·개발 프로젝트입니다. 거대 자원 없이도 작은 모델로 어디까지 갈 수 있는지 실험하는 개인 테스트베드이며, 다음 다섯 가지를 설계 철학으로 둡니다.

> **이름의 유래** — 피에로(Pierrot)는 원래 무언극에서 **남을 따라 하고 흉내 내는** 광대 캐릭터입니다. 기존 연구의 좋은 점을 따라 재현·결합한다는 이 프로젝트의 첫 번째 철학(MimiC)과 맞닿아 있어 이 이름으로 정했습니다.

1. **차용과 재현 우선 (MimiC)** — 새로운 알고리즘을 직접 시도하기도 하지만, 기본 방향은 기존 연구들의 검증된 장점을 골라 **재현·결합**하는 것입니다.
2. **저비용 소형 모델 지향** — 학습 방법 · 모델 구조 · 수렴 전략 모든 단계에서 **메모리를 최소화**합니다. 적은 자원으로 가능성을 빠르게 검증하는 실험 장치입니다.
3. **상업 모델에 근접하는 품질 목표 지향** — 가능성이 확인되면, 작은 모델이더라도 **기존 상업용 모델에 근접한 품질**을 목표로 점진적으로 키워 나갑니다.
4. **개인적 호기심의 실험장** — 무엇보다, 만드는 사람의 호기심을 푸는 공간입니다.
5. **오픈소스 지향** — 이 프로젝트는 기본적으로 오픈소스를 지향합니다.

아래 샘플 그리드를 보면, **0.8B 규모치고는** 꽤 준수한 생성 품질을 보여줍니다.

<p align="center">
  <img src="docs/baseprior_grid_step02370000_20260616.png" width="100%" alt="PIERROT 0.8B 생성 샘플 그리드"/>
  <br/>
  <sub>PIERROT 0.8B · step 2.37M · 1024² · 28 steps — 다양한 프롬프트 생성 샘플</sub>
</p>

## 💬 잡담

딥러닝은 오래 해왔지만 Diffusion 은 약 1.5년 전에 처음 접했습니다. 그동안 LoRA 기반 도메인 특화 학습만 반복하다 보니 조금 지겨웠고, 그사이 쏟아지는 연구들을 보며 "미국·중국 회사들은 꾸준히 좋은 모델을 내는데 왜 한국 회사들은 잘 안 할까", "스크래치(from scratch) 학습이 정말 그렇게 어려울까" 가 궁금해졌습니다. 무엇보다, 나만의 모델을 바닥부터 직접 만들어 보고 싶었습니다.

제약은 많았습니다 — 바쁜 회사 일, 따로 내야 하는 개인 시간, 그리고 가장 큰 벽인 비용(고비용 서버를 개인이 감당). 그래도 막연히 논문만 보고 있기보다, 직접 실행해 부딪혀 보자는 생각으로 시작한 프로젝트입니다.

아직 진행 중이지만, 지금까지의 결론은 — **GPU 와 학습셋만 갖춰진다면**(쉬운 조건은 아니지만) 제가 좋아하는 **[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** 같은 작고 실용적인 고성능 모델도 충분히 가능하겠다는 생각·확신이 조금 들었습니다.

소심한 성격:)이니 나쁜말보단 좋은 말로 응원 부탁드립니다.

## 📰 News

- 2026-07-15 — 📚 최신 논문 리뷰: [DAR: Rethinking Cross-Layer Information Routing in Diffusion Transformers](https://github.com/Pierrot-vision/Reading-Papers/blob/main/Diffusion/PAPER_DAR.md)
- 2026-07-09 — 🖼️ step별 샘플 관찰 기록 공개 — 0.8B ([LAB/0.8b_training_review.md](LAB/0.8b_training_review.md)) · 1.6B ([LAB/1.6b_training_review.md](LAB/1.6b_training_review.md))
- 2026-07-02 — 🧪 LAB(실험 일기) 개설 — SFT 실험·운영 노트 공개 ([LAB/SFT.md](LAB/SFT.md))
- 2026-06-16 — beta-v2370:base 0.8b (v1) 추론용 체크포인트 공개 ([다운로드 ↓](#-체크포인트))
- 2026-06-16 — PIERROT 추론 전용 패키지 공개 (코드 + 의존성 + 문서 + CC BY-NC-SA 4.0)

## ✨ 특징

- **순수 PyTorch** — 학습 의존성 0. `torch` + `diffusers`(VAE) + `transformers`(text encoder) 만 있으면 동작.
- **자기 완결적(self-contained)** — 모델 · 스케줄러 · 파이프라인 · 토크나이저가 한 패키지 안에 모두 포함.
- **4D RoPE+Hybrid 트랜스포머** — (t, h, w, l) 좌표계로 메인 이미지 + 다중 참조 + 어순을 한 번에 인코딩.
- **Flow Matching 학습/추론** — `x_prediction` / `velocity` 두 모드, 해상도 적응형 `snr_shift` 스케줄.
- **학습 모델과 bit-exact** — 학습에 쓰인 가중치를 그대로 strict 로드, 출력이 학습 그래프와 비트 단위로 동일.

## 🧬 모델 스펙

| 항목 | 값 |
|---|---|
| 파라미터 | **0.857B** (config preset `0.8b`) |
| 구조 | Hybrid DiT — depth **16** |
| hidden / heads | 1792 / 28 (head_dim 64) |
| GQA K/V heads | 4 |
| AdaLN embed dim | 256 |
| 위치 인코딩 | 4D RoPE (t, h, w, l), axes_dim [16, 16, 16, 16], θ=2000 |
| latent | 32ch, patch_size 2 (FLUX.2 VAE) |
| 해상도 | 1024² (multi-aspect) |
| objective / schedule | Flow Matching `x_prediction` / `snr_shift` |
| text encoder | `Qwen/Qwen3-4B` (hidden layers 9·18·27 concat → 7680-d) |
| VAE | `black-forest-labs/FLUX.2-small-decoder` |
| precision | bf16 |

## 🧠 주요 알고리즘

| 알고리즘 | 핵심 역할 | 비고 / 영감 |
|---|---|---|
| **4D RoPE** (t, h, w, l) | 메인 이미지 · 다중 참조 · 텍스트 어순을 한 좌표계로 회전 위치 인코딩 | FLUX.2 |
| **Hybrid 블록** | 앞쪽 N개 = 양방향 `PIERROTDualBlock`(MMDiT식, 텍스트도 Q 생성), 나머지 = 비대칭 `PIERROTBlock`(텍스트 KV-only) | MMDiT · PRX |
| **Flow Matching** | 학습/추론 목적함수. `x_prediction`(x₀ 직접 예측 후 velocity 환산) / `velocity` 두 모드 | Rectified Flow |
| **snr_shift 스케줄** | 이미지 토큰 수 기반 μ 로 timestep 을 해상도 적응형 재분배 (어려운 t≈1 구간에 step 집중) | SANA · FLUX |
| **GQA** (`n_kv_heads=4`) | K/V head 공유로 attention 메모리·연산 절감 | LLaMA-2 |
| **AdaLN-Zero** (4-param) | timestep 조건부 scale/gate 변조. shift 제거로 modulation 파라미터 절감 | DiT |
| **Sandwich-Norm + tanh-gate + RMSNorm** | 깊은 모델 bf16 잔차 진폭 안정화 | Z-Image |
| **CFG** (classifier-free guidance) | `guidance_scale` 로 프롬프트 충실도 제어 | — |
| **chi_prompt** (옵션) | text encoder 입력 prefix 로 프롬프트 의미 확장 | SANA  |
| **Quote char-level 토크나이징** | 따옴표(`"..."`) 안 텍스트를 글자 단위로 토큰화 → scene-text 철자 정확도 ↑ | LongCat |

## 💾 메모리 절약 기법

저비용·소형을 지향하므로 모델 구조부터 학습까지 메모리를 아끼는 기법을 적극 사용합니다.

| 기법 | 적용 단계 | 메모리 절약 효과 |
|---|---|---|
| **소형 모델 (0.857B)** | 구조 | 상업용 대비 수~수십분의 1 규모로 파라미터 자체 최소화 |
| **GQA** (`n_kv_heads=4`) | 구조 · 학습/추론 | K/V head 28→4 공유로 attention K/V 메모리·연산 ↓ |
| **AdaLN-Zero 4-param + `adaln_embed_dim=256`** | 구조 | shift 제거 + modulation 입력 차원 cap → modulation 파라미터 대폭 ↓ |
| **RMSNorm** | 구조 | LayerNorm 대비 통계·연산 단순화 |
| **patch_size 2 + 32ch latent** (FLUX.2 VAE) | 구조 | 토큰 수 ↓ → attention/activation 메모리 ↓ |
| **SDPA** (FlashAttention / cuDNN) | 학습/추론 | O(N) 메모리 attention 커널 |
| **bf16 mixed precision** | 학습/추론 | 절반 정밀도로 weight·activation 메모리 ↓ |
| **TREAD 토큰 라우팅** (0.75) | 학습 | 중간 블록에 토큰 일부만 통과 → activation 메모리·step 비용 ↓ |
| **Muon optimizer** | 학습 | 2D weight optimizer state 경량화 |
| **precompute latent-only** | 학습 | 학습 중 VAE 미사용(latent 만 로드) → GPU 메모리·연산 ↓ |
| **gradient accumulation** | 학습 | 작은 batch 로 큰 effective batch (peak 메모리 ↓) |

> 특히 **GQA(`n_kv_heads=4`) + adaLN cap(`adaln_embed_dim=256`)** 두 기법만으로, 같은 아키텍처(hidden 1792 · depth 16 · 28 head)의 raw MHA **1.46B → 0.857B (약 41% 절감)** 입니다 (configs.py 명시). 즉 0.857B 는 "효율화된 1.46B 급" 모델입니다.

## ⚡ 학습 수렴 가속

적은 예산으로 빠르게 수렴시키기 위해, 표현 정렬 · perceptual loss · optimizer 등을 함께 사용합니다.

| 기법 | 역할 | 비고 / 영감 |
|---|---|---|
| **REPA** (DINOv3 표현 정렬) | 트랜스포머 hidden state 를 사전학습 비전 인코더(DINOv3) 표현에 정렬 → 의미 표현 학습 가속 | 초반 burn-in 집중 후 제거 (REPA) |
| **LPIPS** | 디코드 이미지에 perceptual loss → 픽셀·질감 수렴 가속 | LPIPS |
| **Perceptual-DINO** | DINO feature 기반 semantic perceptual loss | — |
| **x_prediction** | x₀ 직접 예측 출력 계약 → LPIPS / P-DINO 같은 픽셀공간 loss 적용 가능 | — |
| **Muon optimizer** | 2D weight 직교화 업데이트로 step 당 수렴 가속 | Muon |
| **EMA** | 가중치 지수이동평균 → 안정화 + 일반화 향상 | — |
| **TREAD 토큰 라우팅** | 중간 블록 토큰 일부만 통과 → 학습 wall-clock ~50% 단축 | TREAD |

## 📦 구조

```
PIERROT/
├── sample.py              # CLI 진입점  (python -m PIERROT.sample)
├── infer.sh               # 단일 프롬프트 추론 wrapper
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

## 🚀 Install

```bash
# 1) 클론 (패키지 폴더명은 PIERROT 로)
git clone https://github.com/Pierrot-vision/Pierrot PIERROT

# 2) conda 환경 생성 + 활성화 (Python 3.12)
conda create -n pierrot python=3.12 -y
conda activate pierrot

# 3) 의존성 설치
pip install -r PIERROT/requirements.txt

# 4) PIERROT 의 상위 디렉토리에서 실행 (python -m PIERROT.sample)
cd "$(dirname PIERROT)"
```

> 텍스트 인코더(`Qwen/Qwen3-4B`)와 VAE(`black-forest-labs/FLUX.2-small-decoder`)는 첫 실행 시 Hugging Face 에서 자동 다운로드됩니다.

## 📥 체크포인트

| 모델 | 종류 | 파일 | 권장 해제 위치 | 업데이트 | 다운로드 |
|---|---|---|---|---|---|
| **beta-v2370 base 0.8b (1차)** | 추론용 (model_only) | `model_base.zip` → `model.pt` | `checkpoints/0.8b_base/model.pt` | 2026-06-16 | [Google Drive](https://drive.google.com/file/d/1HX5zMDnRStYWHlU4WyROoU4vCY8L51MA/view?usp=sharing) |

> 압축을 풀어 나온 `model.pt` 를 `--ckpt` 에 지정하면 됩니다 (CLI 예시는 아래 참고).
> Fine-tuning 용 체크포인트 (optimizer / scheduler / EMA 포함) 도 공개 예정이지만, 현재 Google Drive 용량 부족으로 보류 중입니다 (대체 저장소 마련 중).
> 모델은 최신 버전으로 수시 업데이트될 수 있습니다.

## 🎨 추론

### CLI

```bash
cd <PIERROT 의 상위 디렉토리>

CUDA_VISIBLE_DEVICES=0 python -m PIERROT.sample \
    --ckpt   checkpoints/0.8b_base/model.pt \
    --prompt "a red apple on a wooden table" \
    --output PIERROT/results/apple.png \
    --steps 28 --seed 42
```

또는 단일 프롬프트 wrapper (출력은 `PIERROT/results/`):

```bash
# env 로 override: CKPT(필수) / PROMPT / OUTPUT / GPU / SEED / STEPS / CFG / PYTHON_BIN
CKPT=checkpoints/0.8b_base/model.pt \
PROMPT="a red apple on a wooden table" \
GPU=0 bash PIERROT/infer.sh
```

### Python API

```python
import torch
from PIERROT.model import PIERROT
from PIERROT.model.configs import CONFIG_PRESETS
from PIERROT.pipeline import PIERROTPipeline

# 모델 빌드 + 체크포인트 로드는 sample.py 의 build_model() 참고
model = PIERROT(CONFIG_PRESETS["0.8b"]).eval()
# ... vae / text_encoder / tokenizer 준비 후 ...
pipe  = PIERROTPipeline(model=model, text_encoder=enc, tokenizer=tok, vae=vae)
image = pipe(prompt="a red apple on a wooden table",
             num_inference_steps=28, guidance_scale=4.0)["images"][0]
```

전체 옵션은 `python -m PIERROT.sample --help` 를 참고하세요.

## 🔧 알림

- 공개된 모델은 **상업용 모델 수준의 성능에는 미치지 못합니다.** 그런 성능이 필요하시면 대형 기업의 공개 모델을 이용하시길 권합니다. 생성 가능한 범위는 아래 **공개 학습셋**에 의존한다고 보시면 됩니다.
- 이 패키지는 현재 공개하지 않는 PIERROT_LAB에서 실험한 추론 코드입니다.
- 현재 학습 코드 공개할 계획을 가지고 있지 않습니다. (향후 고민)
- 1인 프로젝트라 비용에 대한 문제가 존재하여 진행속도가 느릴수 있습니다.
- **참여·조언·교류·채용 등 관련 모든 문의는 언제나 환영입니다.**
- **관련 GPU·데이터셋등의 지원은 언제나·항상 정중히 감사하게 기다리고 있습니다.:)**  

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
- [x] 다중 참조(multi-reference) 입력 지원 → 코드만 완료
- [x] 체크포인트 공개
- [x] Post-Training : SFT → 0.8b 완료 → [SFT.md](LAB/SFT.md) 참조
- [x] step별 샘플 관찰 기록 → [0.8b_training_review.md](LAB/0.8b_training_review.md) · [1.6b_training_review.md](LAB/1.6b_training_review.md) 참조

**진행 / 예정**

- [ ] Post-Training : DPO 같은 알고리즘 개발 
- [ ] depth growth 스케일업 (→ 1.6B) → 학습 진행중
- [ ] depth growth 스케일업 (→ 3.2B) → 예정
- [ ] Turbo 버전 모델 개발 - few-step & cfg distillation 
- [ ] 정량 벤치마크 (GenEval / DPG-Bench 등)
- [ ] Edit 모델 개발 → Domain-Speicific 영역에서 계획중 
- [ ] Next (새로운 알고리즘) 모델 개발 → 코드만 완료 → 리소스 부족으로 실행 무
- [ ] LoRA 모델 추가

## 🤗 Reference

- PIERROT 는 기존 연구들의 좋은 점들을 재현·결합하여 만들어집니다.
- 저는 이 연구들에 대해 항상 감사한 마음을 가집니다.

📚 **참고한 논문 전체(리뷰 포함) 리스트 → [Pierrot-vision/Reading-Papers](https://github.com/Pierrot-vision/Reading-Papers)** 

## 📄 라이센스

이 프로젝트는 **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) 를 따릅니다 — 자세한 내용은 [LICENSE](LICENSE) 참고.

- ✅ **연구 · 학술 · 교육 · 개인 실험** 목적의 사용 / 수정 / 재배포 허용
- ❌ **상업적 사용 금지** — 유료 서비스 · 제품 · API · 상업 모델 개발 등 (모델 가중치 · 생성 결과물 포함)
- 재배포 · 파생물은 **출처 표기 + 동일 라이센스(CC BY-NC-SA 4.0) 유지** 필요 (저작자 · 라이센스 링크 · 변경 사항 명시, [Pierrot-vision/Pierrot](https://github.com/Pierrot-vision/Pierrot))
- 학습에 사용된 외부 데이터셋 · 모델 · 라이브러리는 **각자의 라이센스**를 따릅니다
- 상업적 이용 문의는 메인테이너에게 연락해 주세요

## 📮 문의

- [메일](mailto:peternara@naver.com) 또는 [GitHub Issue](https://github.com/Pierrot-vision/Pierrot/issues) 를 통해 관련 질문·문의 부탁드립니다. 대답할수 있는 내용이라면 성실이 답변드리겠습니다.
- 참고로, 이미 GitHub(README · 코드 · 문서)에 있는 내용을 다시 문의하시면 답을 드리지 못할 수 있는 점 양해 부탁드립니다.
