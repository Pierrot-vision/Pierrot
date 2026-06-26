<p align="center">
  <img src="docs/pierrot-banner.png" width="100%" alt="PIERROT banner"/>
</p>

<h1 align="center">🎭 PIERROT</h1>

<p align="center">
  <b>A solo image-generation model development project</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/model-0.857B-success.svg" alt="params"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg" alt="license"/>
</p>

<p align="center">
  <a href="README.md">한국어</a> | <b>English</b>
</p>

---

## 💡 Introduction

**PIERROT** is a solo **text-to-image (T2I)** generation research & development project. It is a personal testbed for exploring how far a *small* model can go without massive resources, guided by five design principles:

> **Origin of the name** — Pierrot is a pantomime clown character who **mimics and imitates others**. That resonates with the project's first principle (MimiC) — reproducing and recombining the strengths of existing research — so the name was chosen.

1. **Borrow & reproduce first (MimiC)** — While new algorithms are tried too, the default direction is to pick the proven strengths of existing research and **reproduce / recombine** them.
2. **Low-cost, small-model oriented** — **Minimize memory** at every stage (training method, model structure, convergence strategy). A device for quickly validating feasibility with limited resources.
3. **Aiming for near-commercial quality** — Once feasibility is confirmed, gradually scale up toward **quality close to commercial models**, even with a small model.
4. **A playground for personal curiosity** — Above all, a space to satisfy the maker's curiosity.
5. **Open-source oriented** — this project is open-source by default.

As the sample grid below shows, the quality is quite decent **for a 0.8B-scale model**.

<p align="center">
  <img src="docs/baseprior_grid_20260616.png" width="100%" alt="PIERROT 0.8B sample grid"/>
  <br/>
  <sub>PIERROT 0.8B · 1024² · 28 steps — samples from diverse prompts</sub>
</p>

## 📰 News

- 2026-06-19 — 📚 Latest paper review: [i1: A Recipe for Text-to-Image Diffusion from Public Materials](https://github.com/Pierrot-vision/Reading-Papers/blob/main/Diffusion/PAPER_i1.md)
- 2026-06-16 — `beta-v2370` base 0.8b (v1) inference checkpoint released ([download ↓](#-checkpoints))
- 2026-06-16 — PIERROT inference-only package released (code + deps + docs + CC BY-NC-SA 4.0)

## ✨ Features

- **Pure PyTorch** — zero training dependencies. Runs with just `torch` + `diffusers` (VAE) + `transformers` (text encoder).
- **Self-contained** — model, scheduler, pipeline, and tokenizer are all in one package.
- **4D RoPE + Hybrid transformer** — encodes the main image + multiple references + token order in a single (t, h, w, l) coordinate system.
- **Flow Matching (train/infer)** — `x_prediction` / `velocity` modes, resolution-adaptive `snr_shift` schedule.
- **Bit-exact with the training model** — loads training weights strictly; output is bit-for-bit identical to the training graph.

## 🧬 Model Spec

| Item | Value |
|---|---|
| Parameters | **0.857B** (config preset `0.8b`) |
| Architecture | Hybrid DiT — depth **16** |
| hidden / heads | 1792 / 28 (head_dim 64) |
| GQA K/V heads | 4 |
| AdaLN embed dim | 256 |
| Positional encoding | 4D RoPE (t, h, w, l), axes_dim [16, 16, 16, 16], θ=2000 |
| Latent | 32ch, patch_size 2 (FLUX.2 VAE) |
| Resolution | 1024² (multi-aspect) |
| Objective / schedule | Flow Matching `x_prediction` / `snr_shift` |
| Text encoder | `Qwen/Qwen3-4B` (hidden layers 9·18·27 concat → 7680-d) |
| VAE | `black-forest-labs/FLUX.2-small-decoder` |
| Precision | bf16 |

## 🧠 Core Algorithms

| Algorithm | Role | Notes / inspiration |
|---|---|---|
| **4D RoPE** (t, h, w, l) | Rotary positional encoding of main image · multi-ref · text order in one coordinate system | extends FLUX.2 RoPE convention |
| **Hybrid blocks** | First N = bidirectional `PIERROTDualBlock` (MMDiT-style, text also produces Q); rest = asymmetric `PIERROTBlock` (text KV-only) | MMDiT · PRX |
| **Flow Matching** | Train/infer objective. `x_prediction` (predict x₀ then convert to velocity) / `velocity` | Rectified Flow |
| **snr_shift schedule** | Resolution-adaptive timestep redistribution via μ from image token count (concentrates steps near hard t≈1) | SANA · FLUX |
| **GQA** (`n_kv_heads=4`) | Shared K/V heads → lower attention memory/compute | LLaMA-2 |
| **AdaLN-Zero** (4-param) | Timestep-conditioned scale/gate modulation; shift removed to cut modulation params | DiT |
| **Sandwich-Norm + tanh-gate + RMSNorm** | Stabilizes bf16 residual magnitude in deep models | Z-Image |
| **CFG** (classifier-free guidance) | Controls prompt fidelity via `guidance_scale` | — |
| **chi_prompt** (optional) | Text-encoder input prefix to expand prompt meaning | SANA (Complex Human Instruct) |
| **Quote char-level tokenization** | Char-level tokenization of quoted (`"..."`) text → better scene-text spelling accuracy | LongCat |

## 💾 Memory-Saving Techniques

Being low-cost and small-oriented, memory-saving techniques are used aggressively from model structure to training.

| Technique | Stage | Memory benefit |
|---|---|---|
| **Small model (0.857B)** | structure | Minimizes parameters — a fraction (1/several to 1/tens) of commercial models |
| **GQA** (`n_kv_heads=4`) | structure · train/infer | K/V heads 28→4 → lower attention K/V memory/compute |
| **AdaLN-Zero 4-param + `adaln_embed_dim=256`** | structure | shift removed + modulation input-dim cap → far fewer modulation params |
| **RMSNorm** | structure | simpler stats/compute vs LayerNorm |
| **patch_size 2 + 32ch latent** (FLUX.2 VAE) | structure | fewer tokens → lower attention/activation memory |
| **SDPA** (FlashAttention / cuDNN) | train/infer | O(N)-memory attention kernel |
| **bf16 mixed precision** | train/infer | half precision → lower weight/activation memory |
| **TREAD token routing** (0.75) | train | only part of tokens pass mid blocks → lower activation memory & step cost |
| **Muon optimizer** | train | lighter optimizer state for 2D weights |
| **precompute latent-only** | train | no VAE during training (load latents only) → lower GPU memory/compute |
| **gradient accumulation** | train | large effective batch from small batch (lower peak memory) |

> In particular, **GQA (`n_kv_heads=4`) + adaLN cap (`adaln_embed_dim=256`)** alone shrink the same architecture (hidden 1792 · depth 16 · 28 heads) from raw MHA **1.46B → 0.857B (~41% smaller)** (noted in configs.py). In other words, the 0.857B model is an "efficiency-optimized 1.46B-class" model.

## ⚡ Convergence Acceleration

To converge fast on a small budget, representation alignment · perceptual losses · optimizers are combined.

| Technique | Role | Notes / inspiration |
|---|---|---|
| **REPA** (DINOv3 alignment) | Aligns transformer hidden states to a pretrained vision encoder (DINOv3) → faster semantic-representation learning | early burn-in then removed (REPA) |
| **LPIPS** | Perceptual loss on decoded image → faster pixel/texture convergence | LPIPS |
| **Perceptual-DINO** | Semantic perceptual loss on DINO features | — |
| **x_prediction** | Predict-x₀ output contract → enables pixel-space losses like LPIPS / P-DINO | — |
| **Muon optimizer** | Orthogonalized updates for 2D weights → faster per-step convergence | Muon |
| **EMA** | Exponential moving average of weights → stability + generalization | — |
| **TREAD token routing** | Only part of tokens pass mid blocks → ~50% training wall-clock cut | TREAD |

## 📦 Structure

```
PIERROT/
├── sample.py              # CLI entry point  (python -m PIERROT.sample)
├── infer.sh               # single-prompt inference wrapper
├── docs/                  # banner · sample images
├── results/               # inference output folder
├── model/                 # PIERROT transformer
│   ├── pierrot.py
│   ├── pierrot_modules.py     # 4D RoPE · asymmetric/bidirectional blocks · AdaLN
│   └── configs.py             # CONFIG_PRESETS (tiny / 0.8b / 1.6b)
├── scheduler/             # inference-only schedule + step
│   ├── timestep_schedule.py   # linear / linear_shift / snr_shift
│   └── flow_matching.py       # euler_step · x_prediction_to_velocity
└── pipeline/                  # text→image inference + prompt preprocessing
    ├── sampling.py            # denoise / denoise_cfg
    ├── PIERROTPipeline.py     # user-friendly wrapper
    ├── text_encoding.py       # quote-aware tokenizer
    └── constants.py           # chi_prompt default template
```

## 🚀 Install

```bash
# 1) Clone (name the package folder PIERROT)
git clone https://github.com/Pierrot-vision/Pierrot PIERROT

# 2) Create + activate a conda env (Python 3.12)
conda create -n pierrot python=3.12 -y
conda activate pierrot

# 3) Install dependencies
pip install -r PIERROT/requirements.txt

# 4) Run from PIERROT's parent directory (python -m PIERROT.sample)
cd "$(dirname PIERROT)"
```

> The text encoder (`Qwen/Qwen3-4B`) and VAE (`black-forest-labs/FLUX.2-small-decoder`) are auto-downloaded from Hugging Face on first run.

## 📥 Checkpoints

| Model | Type | File | Unpack to (recommended) | Updated | Download |
|---|---|---|---|---|---|
| **beta-v2370 base 0.8b (v1)** | inference (model_only) | `model_base.zip` → `model.pt` | `checkpoints/0.8b_base/model.pt` | 2026-06-16 | [Google Drive](https://drive.google.com/file/d/1HX5zMDnRStYWHlU4WyROoU4vCY8L51MA/view?usp=sharing) |

> Unzip and pass the resulting `model.pt` to `--ckpt` (see the CLI example below).
> Fine-tuning checkpoints (with optimizer / scheduler / EMA) will be released too, but are on hold due to limited Google Drive storage (working on an alternative).
> The model may be updated to newer versions from time to time.

## 🎨 Inference

### CLI

```bash
cd <parent of PIERROT>

CUDA_VISIBLE_DEVICES=0 python -m PIERROT.sample \
    --ckpt   checkpoints/0.8b_base/model.pt \
    --prompt "a red apple on a wooden table" \
    --output PIERROT/results/apple.png \
    --steps 28 --seed 42
```

Or the single-prompt wrapper (output goes to `PIERROT/results/`):

```bash
# override via env: CKPT(required) / PROMPT / OUTPUT / GPU / SEED / STEPS / CFG / PYTHON_BIN
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

# See sample.py's build_model() for model build + checkpoint loading
model = PIERROT(CONFIG_PRESETS["0.8b"]).eval()
# ... after preparing vae / text_encoder / tokenizer ...
pipe  = PIERROTPipeline(model=model, text_encoder=enc, tokenizer=tok, vae=vae)
image = pipe(prompt="a red apple on a wooden table",
             num_inference_steps=28, guidance_scale=4.0)["images"][0]
```

See `python -m PIERROT.sample --help` for all options.

## 🔧 Notes

- The released model does **not** reach commercial-grade quality. If you need that level, please use a public model from a larger company. What it can generate largely depends on the **public training sets** listed below.
- This package is inference code from the (currently private) PIERROT_LAB experiments.
- There is currently no plan to release the training code. (under consideration)
- As a one-person project, budget constraints mean progress may be slow.
- **All inquiries — participation · advice · hiring — are always welcome.**
- **Support such as GPUs is always, gratefully welcomed. :)**

## 📚 Training Datasets

- PIERROT was trained using the following public datasets. (Thanks!)
- Not all datasets are used in full — some use only a part / cleaned subset.

| Dataset | Public source | Scale (approx.) | Role |
|---|---|---:|---|
| **FLUX-Reason-6M** | [LucasFang/FLUX-Reason-6M](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) | 5.89M | reasoning-rich synthetic caption prior |
| **flux_generated** | [lehduong/flux_generated](https://huggingface.co/datasets/lehduong/flux_generated) | 1.75M | general synthetic diversity |
| **CC12M** (WebDataset) | [pixparse/cc12m-wds](https://huggingface.co/datasets/pixparse/cc12m-wds) | ~3.66M | web-caption diversity |
| **DiffusionDB** | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) | ~1.0M | diffusion-prompt diversity |
| **MONET** (Megalith-10M / COYO) | [jasperai/monet](https://huggingface.co/datasets/jasperai/monet) | ~3.9M | photo-aesthetic prior (latent-only) |
| **AnyWord-3M** (LAION) | [stzhao/AnyWord-3M](https://huggingface.co/datasets/stzhao/AnyWord-3M) | ~2.36M | scene-text / OCR prior |
| **HumanCaption-10M** | [OpenFace-CQUPT/HumanCaption-10M](https://huggingface.co/datasets/OpenFace-CQUPT/HumanCaption-10M) | ~0.75M | human-scene prior |
| **DeepFashion-MultiModal** | [yumingj/DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) | 12.7K | full-body fashion prior |

## 🗺️ Roadmap (To-Do)

**Done**

- [x] 0.8B base model training — 1024² multi-aspect
- [x] Flow Matching (`x_prediction`) + 4D RoPE Hybrid transformer
- [x] Base pretraining on public datasets
- [x] Inference-only package split — zero training dependencies
- [x] CLI / Python API inference pipeline
- [x] Multi-reference input support (code only)

**In progress / planned**

- [ ] Post-Training: SFT → code done → 0.8b in progress
- [ ] Post-Training: algorithms like DPO
- [ ] Depth-growth scale-up (→ 1.6B) → training in progress
- [ ] Depth-growth scale-up (→ 2.2B) → testing in progress
- [ ] Turbo version — few-step & CFG distillation
- [ ] Quantitative benchmarks (GenEval / DPG-Bench, etc.)
- [ ] Edit model → planned for domain-specific use
- [ ] Next (new-algorithm) model → code done → not run (resource-limited)
- [ ] Add LoRA models

## 🤗 Reference

- PIERROT is built by reproducing and recombining the strengths of existing research.
- I am always grateful to these works.

📚 **Full list of referenced papers (with reviews) → [Pierrot-vision/Reading-Papers](https://github.com/Pierrot-vision/Reading-Papers)**

## 📄 License

This project is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) — see [LICENSE](LICENSE) for details.

- ✅ Use / modify / redistribute allowed for **research · academic · educational · personal experimentation**
- ❌ **No commercial use** — paid services · products · APIs · commercial model development, etc. (includes model weights · generated outputs)
- Redistributions / derivatives must use the **same license (CC BY-NC-SA 4.0) + attribution** (credit · license link · indicate changes; [Pierrot-vision/Pierrot](https://github.com/Pierrot-vision/Pierrot))
- External datasets · models · libraries used in training are subject to **their own licenses**
- For commercial licensing, please contact the maintainer

## 📮 Contact

- Please reach out with questions/inquiries via [mail](mailto:peternara@naver.com) or [GitHub Issue](https://github.com/Pierrot-vision/Pierrot/issues).
- Note: if you ask about something already in the GitHub repo (README · code · docs), I may not be able to respond — thanks for understanding.
