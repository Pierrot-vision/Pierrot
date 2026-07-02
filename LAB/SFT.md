# PIERROT SFT 실험 일기와 운영 노트

작성일: 2026-07-02

이 문서는 PIERROT phase2 base 이후 SFT를 실험하면서 남긴 실험 일기이자 운영 노트다. 목적은 단순히 최종 설정만 기록하는 것이 아니라, 왜 그 설정을 선택했는지, 어떤 가설이 맞거나 틀렸는지, 다음에 같은 문제를 만난 사람이 어디부터 확인하면 되는지를 남기는 것이다.

처음의 핵심 질문은 두 가지였다.

- SFT를 할수록 의미는 유지되는데 화질/선명도가 점점 나빠지는 이유는 무엇인가?
- LPIPS/P-DINO를 켠 뒤 전역 blur drift는 완화됐지만, base에서 잘 되던 일부 능력이 SFT에서 약해지는 문제를 어떻게 줄일 수 있는가?

읽는 순서는 다음을 추천한다.

- 빠르게 결론만 보려면 1장과 15~17장을 먼저 본다.
- blur/화질 저하 원인을 보려면 2~6장을 본다.
- 실제 설정값과 데이터 믹스를 보려면 7~10장을 본다.
- 다음 실험을 이어가려면 12~14장, 17~19장을 본다.

이 문서에서 “base”는 phase2 base 학습 checkpoint를, “SFT”는 phase2 이후 benchmark/instruction/binding 성능을 올리기 위한 후속 학습을 뜻한다. “SFT aux”는 원래 SFT에서 쓰던 데이터셋 일부를 base resume 학습에 낮은 비율로 섞어 넣은 보강 데이터를 뜻한다.

## 1. 현재 결론 요약

현재까지 가장 중요한 결론은 다음이다.

- 초기 SFT 품질 저하의 1차 원인은 SFT에서 `use_lpips=False`, `use_pdino=False`로 두고 MSE 계열 main loss만 남긴 것이었다.
- base phase2는 LPIPS와 P-DINO 지각 손실을 켠 상태였고, SFT에서만 이를 끄면서 고주파/detail 유지 신호가 사라졌다.
- LPIPS/P-DINO를 다시 켠 뒤에는 이전처럼 전체 이미지가 계속 흐려지는 blur drift가 뚜렷하게 재현되지 않았다. 이 가설은 현재까지 가장 강하게 지지된다.
- 하지만 LPIPS/P-DINO를 켜도 base에서 잘 되던 일부 prompt/능력이 SFT에서 약해지는 문제는 남았다. 이 문제는 손실 함수만이 아니라 데이터 분포, replay 비율, LR, synthetic tone, 특정 도메인 과노출의 문제다.
- 그래서 현재 방향은 두 갈래다. 첫째, SFT를 할 때는 `lr=3e-5`, `use_repa=False`, `use_lpips=True`, `use_pdino=True`를 기본으로 둔다. 둘째, SFT 데이터셋 일부를 base resume 학습에 먼저 섞어 넣어 benchmark/binding/text 능력을 base prior 안에 천천히 흡수시키는 실험을 한다.
- SFT aux를 base resume에 넣는 현재 분포는 장기 고정 분포가 아니라 warm phase용이다. 10k/20k/30k checkpoint에서 좋아진 능력과 망가진 능력을 같이 보고, 어느 정도 나오면 synthetic/binding/text aux 비율을 줄인다.
- `max_steps=2_000_000`은 거기까지 반드시 학습하겠다는 뜻이 아니라, 미리 크게 잡아 둔 상한이다. 실제 판단은 중간 checkpoint별 샘플 비교로 해야 한다.

## 2. 실험 타임라인

전체 흐름은 대략 다음과 같다.

| 단계 | 관찰 | 당시 의심 | 이후 판단 |
| --- | --- | --- | --- |
| 초기 SFT | 의미는 맞지만 화질이 점점 흐려짐 | LR, EMA, 데이터, loss 모두 의심 | loss 구성이 가장 강한 1차 원인 |
| base와 비교 | base는 LPIPS/P-DINO ON, SFT는 OFF | SFT에서 perceptual signal이 사라짐 | LPIPS/P-DINO를 다시 켜야 함 |
| 외부 레포 확인 | MiniT2I/LongCat은 MSE 중심, PRX는 perceptual wrapper 존재 | "MSE만 쓰면 안 된다"는 단순 결론은 위험 | 모델/공간/LR/데이터/EMA까지 같이 봐야 함 |
| LPIPS/P-DINO 재활성 | `step_00017500`까지 전역 blur drift는 크게 재현되지 않음 | 초기 원인 가설 강화 | 화질 붕괴와 forgetting을 분리해서 봐야 함 |
| 이후 SFT | base에서 잘 되던 일부 prompt가 약해짐 | 데이터 분포 overwrite | replay/base prior/실사 anchor 강화 방향 |
| `phase2_sft_2` | `step_00055000` 이미지에서 특정 style 영향 의심 | Ovis 전체 영향으로 과도하게 추정 | text Ovis와 generic image style 영향은 구분해야 함 |
| base resume aux 실험 | SFT aux를 base resume에 낮은 비율로 흡수 | 갑작스러운 SFT overwrite 완화 | warm phase로 보고 10k~30k에서 줄일지 판단 |

이 타임라인이 중요한 이유는, 중간에 여러 번 가설이 바뀌었기 때문이다. 특히 "MSE만 있으면 무조건 나쁘다"나 "Ovis가 전체 스타일을 망친다"처럼 단순화하면 실제 문제를 놓치기 쉽다.

## 3. 초기 증상

관찰된 초기 증상은 다음과 같았다.

- SFT step이 진행될수록 의미/전역 구조는 크게 무너지지 않음.
- 반면 texture, 피부/머리카락/배경 detail, sharpness, contrast가 점진적으로 감소.
- `model.pt`와 EMA 모두에서 비슷한 방향의 화질 저하가 보임.
- "의미는 맞는데 이미지가 점점 흐려지는" 형태라 단순 prompt failure보다는 objective drift 가능성이 컸다.

이 패턴은 SFT 데이터가 완전히 잘못됐을 때 나타나는 즉시 붕괴와 다르다. 매 step 조금씩 평균화되는 느낌에 가까웠고, 그래서 손실 함수 구성을 우선 의심했다.

여기서 중요한 감각은 "모델이 말을 못 알아듣는 것"이 아니라 "말은 알아듣는데 이미지가 덜 살아 있는 것"이었다. 그래서 prompt alignment 문제보다 objective나 regularization 문제일 가능성이 컸다.

## 4. base와 초기 SFT의 차이

초기 비교에서 중요한 차이는 다음이었다.

| 항목 | base phase2 | 초기 SFT | 판단 |
| --- | --- | --- | --- |
| `use_repa` | False | False | 동일, 1차 원인 아님 |
| `use_lpips` | True | False | 중요한 차이 |
| `use_pdino` | True | False | 중요한 차이 |
| `lpips_weight` | 0.1 | off | base의 detail 유지 신호가 사라짐 |
| `pdino_weight` | 0.01 | off | semantic/structure perceptual 신호가 사라짐 |
| LR | 당시 1e-4 | 당시 1e-4 | 외부 레포와 비교하면 가능하지만, 수렴 ckpt polish에는 공격적 |
| EMA decay | base 상속 0.999 | 0.999 | MiniT2I와 다르지만 직접 원인으로 보기는 약함 |

초기에는 `use_repa`, `use_lpips`, `use_pdino`를 한 묶음처럼 껐던 판단이 문제였다. REPA는 표현 정렬이므로 base에서 이미 saturation됐다고 보고 끄는 논리가 성립하지만, LPIPS/P-DINO는 매 step 화질 유지에 작동하는 regularizer에 가깝다. 따라서 REPA와 LPIPS/P-DINO를 같은 이유로 끄는 것은 부적절했다.

다시 말해, "REPA를 끄자"는 판단은 여전히 괜찮다. 문제가 된 것은 REPA와 함께 LPIPS/P-DINO까지 같이 끈 것이다.

## 5. 왜 MSE 단독이 blur를 만들 수 있는가

PIERROT의 main loss는 `prediction_mode == "x_prediction"`일 때 직접 x0 MSE가 아니라, 예측 x0를 velocity로 환원해 비교한다.

```python
t_b = t.view(-1, 1, 1, 1).clamp(min=0.05)
pred_for_loss = (noisy_latents - pred) / t_b
target_for_loss = (noisy_latents - target) / t_b
loss = F.mse_loss(pred_for_loss.float(), target_for_loss.float())
```

즉 형식은 velocity-space MSE다. 하지만 perceptual signal이 없으면 여전히 ambiguous detail에 대해 평균적인 방향으로 수렴할 수 있다.

고주파 영역은 정답이 하나로 고정되기 어렵다. 머리카락, 피부 texture, background micro-detail, foliage, neon reflection 같은 영역은 여러 plausible solution이 존재한다. MSE 계열 목적함수는 이런 영역에서 평균값을 선호할 수 있고, 이 평균화가 step마다 누적되면 "의미는 맞지만 선명도가 줄어드는" 형태가 된다.

LPIPS/P-DINO는 이 흐름에 제동을 건다.

- LPIPS: VGG/AlexNet feature 거리로 사람 눈에 가까운 detail/texture 차이를 벌점화.
- P-DINO: DINO feature 기반으로 구조/semantic detail 유지.
- REPA: 모델 내부 feature를 외부 representation에 정렬하는 성격이 강해, 후반 SFT에서 계속 켜야 할 이유는 상대적으로 약함.

이 부분은 "MSE가 나쁘다"는 뜻이 아니다. MSE는 diffusion/flow 학습의 기본 손실이고, 많은 좋은 모델이 MSE 계열로 학습된다. 문제는 수렴된 base checkpoint를 좁은 SFT 데이터로 polish할 때, detail을 지켜주는 보조 신호 없이 같은 방향의 gradient를 오래 누적하는 것이다.

## 6. 현재 PIERROT의 perceptual loss 구현상 주의점

현재 train loop에서 LPIPS/P-DINO는 VAE decode 이미지를 쓰지 않는다. 메모리/속도 절약을 위해 latent 앞 3채널을 pseudo-RGB처럼 사용한다.

```python
x0_image = x0_pred.float()
x0_gt = latents.float()

loss += lpips_weight * lpips_loss(
    x0_image[:, :3].clamp(-1, 1) * 0.5 + 0.5,
    x0_gt[:, :3].clamp(-1, 1) * 0.5 + 0.5,
)
```

따라서 이 신호는 "진짜 decoded image perceptual loss"라기보다 latent-space pseudo perceptual regularizer다. 그래도 초기 관찰상 blur drift 제동에는 효과가 있었지만, 다음 한계가 있다.

- 실제 RGB/VAE decode 기준의 texture 품질과 완전히 동일하지 않다.
- LPIPS/P-DINO가 켜져도 dataset distribution drift나 forgetting을 완전히 막지는 못한다.
- 메모리 여유가 충분하다면 언젠가 작은 batch/낮은 주기로 VAE decode perceptual을 ablation할 가치는 있다.

## 7. 외부 레포 비교

### MiniT2I

MiniT2I는 SFT에서도 RGB/pixel-space flow matching MSE 중심으로 보인다. LR은 pretrain과 finetune 모두 `learning_rate: 0.0004`로 동일하게 유지했다. 즉 SFT라고 LR을 낮추지 않았다. EMA decay는 `0.99995`로 PIERROT의 `0.999`보다 훨씬 느리다.

중요한 차이는 MiniT2I의 MSE가 직접 이미지/RGB 공간에 더 가까운 구조라는 점이다. PIERROT처럼 latent pseudo perceptual을 따로 켠 구조와 단순 비교하면 안 된다.

### LongCat-Image

LongCat 공개 설정은 full SFT에서 latent velocity MSE 중심이고, LR은 `1e-5` 수준으로 낮은 편이다. perceptual auxiliary loss가 공개 설정의 핵심으로 드러나지는 않았다.

여기서 중요한 포인트는 "MSE만 쓰면 항상 망한다"가 아니라, MSE만 쓰더라도 LR, 데이터 품질, 모델/latent 설계, EMA, SFT 길이, replay 비율이 같이 맞아야 한다는 것이다.

### PRX

PRX base loss는 MSE이고, algorithm wrapper로 LPIPS/P-DINO/REPA가 붙는 구조다. PRX Part 3 블로그 기준으로는 LPIPS 0.1, DINO perceptual 0.01을 사용한 것으로 해석된다. 1024 stage에서는 REPA를 끄는 패턴이 맞다.

따라서 PIERROT SFT에서 PRX에 더 맞는 조합은 다음이다.

```python
args.use_repa = False
args.use_lpips = True
args.use_pdino = True
args.lpips_weight = 0.1
args.pdino_weight = 0.01
```

즉 PRX와 맞추려면 REPA만 끄고 LPIPS/P-DINO는 유지하는 쪽이 더 타당하다.

## 8. LR 변화와 외부 연구 비교

LR은 이번 SFT 실험에서 계속 헷갈렸던 지점이라 따로 남긴다. 처음에는 외부 레포를 참고해 base와 같은 `1e-4`를 유지했다. 특히 MiniT2I는 pretrain과 finetune 모두 `learning_rate: 0.0004`로 동일하게 두었기 때문에, 우리도 처음에는 SFT라고 LR을 낮추지 않는 선택이 이상하지 않다고 봤다. PRX 기본 설정도 `1e-4` 계열이라 초기 `1e-4` 선택 자체가 이상한 것은 아니었다.

하지만 PIERROT에서 관찰한 현상은 단순히 “학습이 느리다/빠르다”가 아니라, 이미 수렴한 phase2 base 위에 SFT 데이터가 들어오면서 base 능력을 일부 overwrite하는 형태였다. 이 경우 같은 `1e-4`라도 scratch/base 학습 때와 의미가 다르다. base 학습에서는 넓은 분포를 오래 배우는 LR이지만, SFT에서는 좁은 분포가 이미 형성된 prior를 빠르게 밀어낼 수 있다.

그래서 현재 A안에서는 LR을 다음처럼 낮췄다.

```text
초기 SFT: 1e-4
현재 A안: 3e-5
```

외부 레포와 비교하면 다음처럼 정리할 수 있다.

| 레포/실험 | SFT LR 경향 | 해석 | PIERROT에 주는 의미 |
| --- | --- | --- | --- |
| MiniT2I | pretrain과 finetune 모두 `0.0004`로 동일, SFT에서 LR을 낮추지 않음 | RGB/pixel-space FM MSE 중심, EMA decay도 매우 큼 | “같은 LR도 가능하다”는 중요한 근거지만 PIERROT latent SFT에 그대로 복사하면 안 됨 |
| PRX | 기본 LR은 `1e-4` 계열, perceptual wrapper 사용 가능 | PRX식으로 보면 LR보다 LPIPS/P-DINO/REPA 조합이 중요 | PIERROT 초기 `1e-4` 선택의 근거였지만, SFT retention 문제를 보면 낮출 여지가 있음 |
| LongCat-Image | 공개 full SFT LR은 `1e-5` 수준으로 낮은 편 | 수렴 모델 후속 학습에서 보수적인 LR을 선택한 사례 | PIERROT의 `3e-5`는 MiniT2I식 동일 LR과 LongCat식 낮은 LR 사이의 절충안 |
| PIERROT 초기 SFT | `1e-4` | 외부 레포와 맞추려는 선택 | blur 문제와 forgetting을 분리해 보니, retention 관점에서는 다소 공격적 |
| PIERROT 현재 A안 | `3e-5` | overwrite 속도를 줄이는 polish LR | LPIPS/P-DINO ON과 함께 base 능력 보존을 우선하는 설정 |

따라서 결론은 “기존 연구가 항상 SFT LR을 낮춘다”도 아니고, “base와 동일 LR이 정답이다”도 아니다. MiniT2I처럼 SFT에서도 LR을 전혀 낮추지 않은 사례가 있고, LongCat처럼 낮은 LR을 쓰는 사례도 있다. PIERROT에서는 실제 관찰상 base capability retention이 중요해졌기 때문에 `3e-5`로 낮추는 쪽이 더 안전하다고 판단했다.

이 판단은 특히 다음 조건에서 중요하다.

- 이미 phase2 base가 어느 정도 잘 수렴해 있음.
- SFT 데이터가 base 전체 분포보다 훨씬 좁음.
- synthetic/instruction/text/binding 데이터가 특정 능력을 강하게 밀어줌.
- 목표가 scratch 성능 상승이 아니라 base 능력을 보존하면서 약점을 보강하는 것임.

즉 PIERROT의 `3e-5`는 “외부 레포와 달라서 임의로 낮춘 값”이 아니라, 초기 `1e-4` 실험에서 보인 overwrite 위험을 줄이기 위한 보수적 SFT polish LR이다.

## 9. 현재 적용한 A안

현재 `parse_args_sft_0_8b()` 기준 주요 runtime 설정은 다음이다.

```python
args.batch_size = 10
args.lr = 3e-5
args.epochs = 0
args.max_steps = 2_000_000
args.ckpt_every = 2_500

args.ckpt_dir = "/NHNHOME/WORKSPACE/0426030001_A/DEV/STUDY/PIERROT/checkpoints/phase2_sft_2"
args.resume = "latest"
args.init_model = None

args.use_repa = False
args.use_lpips = True
args.use_pdino = True
args.lpips_weight = 0.1
args.pdino_weight = 0.01
args.aux_warmup_steps = 1000
```

의도는 다음이다.

- LR을 `1e-4`에서 `3e-5`로 낮춰 base capability overwrite 속도를 줄인다.
- REPA는 끄되, LPIPS/P-DINO는 base와 동일하게 유지한다.
- 중간 ckpt 재개 시 aux loss가 갑자기 들어가는 충격을 줄이기 위해 `aux_warmup_steps=1000`을 둔다.
- checkpoint 저장 위치를 `phase2_sft_2`로 분리해 base `phase2`와 섞이지 않게 한다.
- `phase2_sft_2`가 비어 있으면 `phase2` 최신 step의 `model.safetensors`를 init weight로 사용하고, 이후에는 `phase2_sft_2`의 latest를 resume한다.

## 10. SFT 데이터 믹스에서 배운 점

이 장은 `phase2_sft_2` SFT 믹스를 조정하면서 얻은 판단을 정리한 것이다. 2026-07-01 이후에는 이 판단을 바탕으로 “SFT aux 일부를 base resume에 먼저 흡수하는 실험”을 추가했다. base resume 운영 규칙은 16~17장에 따로 정리했다.

현재 SFT는 단순 polish set만 쓰는 것이 아니라 base retention replay를 많이 섞은 상태다. 방향은 다음과 같다.

### 강화한 쪽

- `fine_t2i_curated`: 실사/고품질 anchor 강화.
- `human_hq_sft_subset`, `person_full_model`: 실사 human anatomy, full-body 인물 유지.
- `facecaption_base_mix`, `portrait_sft_mix`: face/portrait realism 강화.
- `object_grounding_sft_10k_multicap`: object/instrument/structure anchor 강화.
- `flux_reason_6m`, `journeydb`, `monet_megalith_part1`, `midjourney_v6_recap`, `diffusiondb_part1`: base prior/replay 강화.

### 낮춘 쪽

- `fine_t2i_synthetic`: synthetic aesthetic tone 완화.
- `dalle3_sft`, `sharegpt4o_sft`, `blip3o_60k`: MiniT2I 계열 synthetic/instruction tone 과노출 완화.
- `aesthetic_4k`, `alchemist`: polish/aesthetic 과적합 완화.
- `ovis_outfit_couple_synth`: two-person/outfit binding에는 도움이 되지만, Ovis synth model prior 과주입 위험 때문에 낮춤.

### 유지/특수 목적

- `ovis_image_text_synth`, `ovis_image_text_hard_synth`, `ovis_scene_text_confusable_synth`: text spelling/OCR-like exact text 목적.
- `couple_combined`: two-person/gender/outfit binding forgetting 방지.
- `instrument_relation_subset_clean`: 악기와 사람/행동 relation binding 유지.
- `base_prior_hq_image_sft_300k`: high-res base-prior image anchor. 이것만으로 모든 base forgetting이 해결되지는 않지만, SFT 데이터가 한쪽 스타일로 치우치는 것을 완화하는 역할이 있다.

## 11. 세 이미지 분석 관련 정정

`phase2_sft_2/step_00055000`에서 본 세 이미지:

- `pierrot_step_00055000_ema_couple_20260701.png`
- `pierrot_step_00055000_ema_woman_20260701.png`
- `pierrot_step_00055000_ema_city_20260701.png`

프롬프트는 각각 다음이었다.

```bash
couple="a young man and woman couple smiling and holding hands on a beach"
woman="a portrait of a woman wearing a blue jacket"
city="a busy city street at night with neon signs"
```

중요한 정정:

- 세 이미지를 직접 visual renderer로 보지는 못했다. 로컬 `view_image`가 sandbox `bwrap` 오류로 실패했다.
- 따라서 이전에 "Ovis 계열 전체 영향"이라고 넓게 말한 것은 과도한 추정이었다.
- caption 분포와 dataset 목적을 기준으로 보면, `ovis_image_text_synth`, `ovis_image_text_hard_synth`, `ovis_scene_text_confusable_synth`는 text/OCR 목적이라 세 generic image의 직접 원인으로 보기 어렵다.
- `ovis_outfit_couple_synth`는 couple/two-person/outfit binding에는 일부 영향이 있을 수 있지만, beach/holding-hands prompt 자체와는 강하게 맞지 않는다.

더 타당한 해석은 다음이다.

| 이미지 | 가장 관련 큰 데이터셋 후보 | 이유 |
| --- | --- | --- |
| couple beach | `couple_combined`, `fine_t2i_curated`, `fine_t2i_synthetic` | two-person/couple semantics + beach/photo style |
| woman blue jacket | `person_full_model`, `portrait_sft_mix`, `facecaption_base_mix`, `human_caption_hq` | portrait/face/full-body/person prior |
| city neon night | `fine_t2i_synthetic`, `dalle3_sft`, `base_prior_hq_image_sft_300k`, `midjourney_v6_recap` | neon/night/city synthetic/photo prompt 분포 |

공통 style source는 Ovis 전체라기보다 `fine_t2i_synthetic`, `dalle3_sft`, `midjourney_v6_recap`, `flux_generated` 같은 synthetic/photo-polish 계열일 가능성이 더 크다. 다만 실제 이미지를 눈으로 확인하지 않은 상태의 결론이므로, 최종 판단에는 직접 이미지 첨부나 VLM caption 분석이 필요하다.

## 12. 현재 관찰

LPIPS/P-DINO를 다시 켠 뒤 `step_00017500`까지는 이전처럼 선명도가 계속 깎이는 현상이 뚜렷하게 재현되지 않았다. 이것은 초기 blur drift의 원인이 perceptual loss OFF였다는 가설을 지지한다.

하지만 다음 문제가 남았다.

- base에서 잘 나오던 일부 prompt가 SFT 후 덜 잘 나옴.
- 특정 synthetic/photo-polish tone이 늘어날 수 있음.
- portrait/couple/city 등 일부 도메인은 SFT mix의 강한 dataset에 끌릴 수 있음.
- text/OCR Ovis 계열은 목적이 분명하지만, generic image 품질에는 직접 도움이 되지 않거나 style prior를 섞을 수 있음.

따라서 지금 단계의 문제는 "화질 붕괴"에서 "base capability retention과 SFT target distribution의 균형" 문제로 이동했다.

## 13. 권장 평가 방식

앞으로 SFT checkpoint를 볼 때는 같은 seed/prompt로 base와 SFT를 나란히 비교해야 한다.

추천 비교 축:

- base phase2 latest EMA
- SFT `step_00017500` EMA
- SFT `step_00035000` EMA
- SFT `step_00055000` EMA
- 필요하면 non-EMA도 같이 확인하되, 최종 판단은 EMA 우선

prompt set은 최소한 다음 그룹을 포함해야 한다.

- base에서 잘 나오던 golden prompts
- portrait/person/full-body prompts
- two-person/couple prompts
- city/night/neon prompts
- object/instrument relation prompts
- text rendering prompts
- fine detail/high-frequency prompts

평가 포인트:

- 의미 정합성
- sharpness/detail
- face realism
- anatomy/full-body consistency
- two-person identity separation
- synthetic/aesthetic tone 과잉 여부
- base 대비 못해진 prompt가 어떤 dataset 방향과 연결되는지

## 14. 다음 액션

1. `phase2_sft_2`의 checkpoint별 golden prompt grid를 만든다.
2. base 대비 나빠진 prompt를 유형별로 분류한다.
3. 나빠진 유형이 특정 dataset 목적과 충돌하는지 본다.
4. synthetic/photo-polish tone이 강하면 `fine_t2i_synthetic`, `dalle3_sft`, `flux_generated`, `midjourney_v6_recap` 쪽을 더 줄이거나 curated/base prior를 늘린다.
5. person/couple이 약하면 `person_full_model`, `couple_combined`, `human_hq_sft_subset`, `facecaption_base_mix`를 조정한다.
6. text만 좋아지고 generic image가 나빠지면 Ovis text 계열 총량을 줄이거나 text 전용 SFT branch로 분리하는 것을 검토한다.
7. LPIPS/P-DINO는 당분간 유지한다. 끄는 ablation은 blur drift 재발 여부를 확인할 목적일 때만 짧게 수행한다.

## 15. 현재 가장 중요한 판단

지금은 `base_prior_hq_image_sft_300k` 하나로 해결될 문제라기보다, SFT 전체 mix의 균형 문제다. `base_prior_hq_image_sft_300k`는 high-res base prior anchor로 유용하지만, 특정 base capability를 모두 커버하지 않는다.

따라서 현재 가장 안전한 방향은 다음이다.

- LR은 `3e-5` 유지.
- REPA는 OFF 유지.
- LPIPS/P-DINO는 ON 유지.
- EMA 기준으로 평가.
- base golden prompts를 checkpoint마다 반드시 회귀 테스트.
- SFT target 데이터와 base replay 데이터의 균형을 prompt failure 유형별로 조정.

한 줄 결론:

> 초기의 "SFT할수록 화질이 점점 나빠지는 문제"는 LPIPS/P-DINO OFF가 가장 강한 원인이었고, 현재 남은 문제는 SFT 데이터 분포가 base 능력을 일부 overwrite하는 retention 문제다.


## 16. 2026-07-01 결정: SFT aux를 base resume에 흡수하는 실험

SFT 자체를 완전히 버리는 것이 아니라, SFT가 맡고 있던 일부 benchmark/binding/text/realism 보강 역할을 base 학습 단계에 낮은 비율로 흡수하는 실험을 추가했다. 의도는 다음이다.

- SFT 후반에 좁은 데이터 분포가 base prior를 갑자기 overwrite하는 것을 줄인다.
- text rendering, multi-object binding, two-person binding, object grounding 같은 benchmark 성능 신호를 base prior 안에 더 천천히 섞는다.
- 기존 phase2 base checkpoint를 resume하여 짧은 warm phase로 흡수시키고, 어느 정도 성능이 나오면 SFT aux 비율을 다시 낮춘다.

따라서 현재 전략은 다음처럼 정리된다.

```text
base resume warm phase:
  broad base prior + SFT aux를 비교적 강하게 섞어 빠르게 약점 보강

after warm phase:
  benchmark 신호가 어느 정도 나오면 synthetic/binding/text aux를 낮춰 base prior 안정화

final SFT:
  필요할 경우 짧고 좁은 benchmark patch 용도로만 사용
```

이 실험은 장기 base 분포를 확정한 것이 아니라, 이전 base 모델을 이어서 학습할 때 초반 10k~30k step 정도의 교정용 curriculum으로 보는 것이 맞다.

## 17. SFT aux warm phase 운영 규칙

현재 분포는 장기 고정 분포라기보다 초기 교정용 warm phase다. 이전 base 모델을 resume할 때 SFT aux를 조금 더 강하게 넣는 것은 타당하지만, 계속 오래 유지하면 synthetic tone이나 prompt-local regression이 base 쪽으로 흡수될 수 있다.

권장 운영은 다음이다.

```text
10k step:
  benchmark 신호가 실제로 움직이는지 확인

20k step:
  text/binding/object 능력 상승 여부 확인

30k step:
  SFT aux를 계속 유지할지, 줄일지 결정
```

줄일 때의 우선순위는 다음이다.

```text
비실사/합성 tone 증가:
  blip3o_60k, fine_t2i_synthetic, dalle3_sft 먼저 완화

text는 좋아졌지만 generic image가 나빠짐:
  ovis_image_text_synth, ovis_image_text_hard_synth, ovis_scene_text_confusable_synth 완화

multi-person/binding이 충분히 올라옴:
  ovis_outfit_couple_synth, couple_combined 완화

실사 anchor가 충분히 안정됨:
  fine_t2i_curated 7% -> 5% 정도로 완화
```

현재 기준의 보수적 안정화 후보는 다음이다.

```text
blip3o_60k:                  5.45% -> 4.0~4.5%
ovis_outfit_couple_synth:    5.45% -> 3.5~4.0%
ovis_scene_text_confusable:  3.27% -> 1.5~2.0%
fine_t2i_synthetic:          3.28% -> 2.0%
dalle3_sft:                  2.18% -> 1.0~1.5%
fine_t2i_curated:            7.00% -> 5.0%
```

단, `fine_t2i_curated`는 현재 실사 anchor 역할이 크므로 가장 마지막에 줄이는 것이 안전하다.

## 18. Prompt distribution 관련 관찰

SFT/base-aux 문제를 볼 때 dataset 비율만큼 중요한 것이 prompt 문체다. 실제 샘플 확인 결과, 현재 학습셋의 안정적인 caption 문체는 대체로 다음과 같다.

```text
A photograph of ...
A close portrait of ...
The exact visible text "..." is clearly displayed ...
a [object] with visible parts, centered composition, realistic lighting
```

반대로 너무 긴 benchmark instruction 문장이나 `premium`, `polished`, `refined`, `gallery-like`, `visual poetry` 같은 추상 품질어는 현재 모델에서 안정적으로 작동하지 않을 수 있다. 일부 prompt에서는 `photorealistic` 같은 단어도 실제 사진 anchor가 아니라 synthetic-realistic attractor처럼 작동했다.

따라서 평가 prompt를 만들 때는 다음 원칙을 둔다.

- 너무 짧게 줄여 subject/structure anchor를 잃지 않는다.
- 긴 instruction 문장은 제거한다.
- subject, object parts, pose, material, lighting, composition, exact visible text는 유지한다.
- text prompt는 `The exact visible text "..."` 형태를 우선 사용한다.
- object/instrument prompt는 `visible parts`, `centered composition`, `realistic object structure` 같은 구조 anchor를 넣는다.

이 관찰은 dataset mix와 별개의 추론 prompt 안정성 문제이므로, checkpoint 비교 시 동일 seed와 함께 prompt variant A/B도 남겨야 한다.


## 19. 이 실험 일기에서 가져갈 교훈

이 실험에서 가장 크게 배운 점은 다음이다.

1. SFT에서 품질이 흐려질 때는 데이터셋만 의심하지 말고 loss 구성을 먼저 확인한다. 특히 base에서 켰던 perceptual regularizer를 SFT에서 껐다면 blur drift가 생길 수 있다.
2. SFT는 benchmark 성능을 올리는 데 유용하지만, 너무 많은 역할을 맡기면 작은 base 재학습처럼 행동한다. 이 경우 base에서 잘 되던 prompt가 일부 망가질 수 있다.
3. 좋은 이미지를 가진 데이터셋이라도 caption 분포가 맞지 않으면 prompt token 의미가 흔들릴 수 있다. `photorealistic`, `premium`, `polished` 같은 단어도 항상 좋은 방향으로 작동하지 않는다.
4. SFT 데이터셋을 base resume에 섞는 방식은 갑작스러운 overwrite를 줄이는 대안이 될 수 있다. 다만 이것도 장기 고정 분포가 아니라 warm phase로 보고, 좋아진 뒤에는 줄여야 한다.
5. 평가는 반드시 같은 seed, 같은 prompt, 같은 checkpoint 간격으로 해야 한다. 평균적으로 좋아져도 특정 golden prompt가 망가질 수 있기 때문이다.
6. 최종 결론을 너무 빨리 내리지 않는다. 이 문서의 여러 판단도 checkpoint와 샘플을 보면서 계속 수정된 것이다. 따라서 새로운 실험을 이어갈 때는 이 문서를 정답지가 아니라, 실패를 줄이기 위한 지도처럼 쓰는 것이 맞다.
