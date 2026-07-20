# 글자는 배웠지만 서체는 배우지 않았다 — 타이포그래피 프로브

작성일: 2026-07-20

PIERROT의 장면 텍스트(scene text) 학습셋 `ovis_image_text_synth`(129,991장)를 분석한 뒤, **학습셋에 없는 것만 골라** 1.6B 모델에 물어본 기록이다. 같은 형식의 인물 진단은 [person_coverage_probe.md](person_coverage_probe.md)에 있다.

## 1. 학습셋의 공백

캡션 7,580건을 슬롯 단위로 셌다. 캡션 형식은 다음 한 줄로 고정돼 있다.

> `A {장면} with the exact visible text "{텍스트}".`

| 축 | 학습셋 실태 |
| --- | --- |
| 텍스트 | 고유 문구 **238종**, 고유 단어 **106개**. `PIERROT` 38.2% · `GALLERY` 20.5% · `CLASSIC` 8.6% · `MAGAZINE` 7.8% · `NEW ARRIVAL` 4.9% |
| 장면 | **단 9종** (영화포스터·타이틀카드·로고·3D렌더·병라벨·책표지·네온간판·스티커·카페간판) |
| **서체·폰트** | `font` · `serif` · `sans-serif` · `typeface` · `typography` · `italic` · `script` · `handwritten` · `calligraphy` · `chrome` · `gradient` · `outlined` · `condensed` **전부 0%**. `bold`만 0.47% |

세 번째 줄이 핵심이다. **이 데이터셋은 "무슨 글자를 쓸지"만 가르치고 "어떤 서체로 쓸지"는 한 번도 가르치지 않았다.** `caption_short` / `caption_medium` / `caption_long` 세 변형을 다 뒤져도 서체 관련 단어가 없다.

## 2. 어떻게 물어봤나

학습 캡션 형식을 그대로 따르되 세 가지를 학습셋 밖으로 두었다.

- **서체 서술을 넣었다** — 학습셋 0%인 12종 폰트와 6종 표현 방식
- **새 텍스트를 썼다** — 학습셋 106개 단어와 **충돌 0건** (자동 검사로 확인)
- **새 장면을 썼다** — 9종 밖의 매체 (레터프레스·타일 모자이크·자수 캡·나무 표지판 등)

전문은 [ovis_probe_prompts.json](../ovis_probe_prompts.json)에 있다.

| 항목 | 값 |
| --- | --- |
| 모델 | PIERROT 1.6B — phase3 step **935k** (EMA) |
| 해상도 / step / CFG | 1024 × 1024 / 28 / 4.0 |
| seed | 42 (프롬프트마다 고정) |
| chi_prompt | OFF |

| 축 | id | 요청한 서체 |
| --- | --- | --- |
| 폰트 종류 | 1–12 | serif · geometric sans · script · handwritten · calligraphy · blackletter · condensed · monospace · stencil · pixel · bubble · italic |
| 표현 방식 | 13–18 | chrome · gradient · outline · emboss · embroidery · carved |
| 대소문자·길이 | 19–22 | 대소문자 혼합 · 긴 두 단어 · 숫자 포함 · 2행 배치 |
| 신규 장면 | 23–24 | 타일 모자이크 · 나무 스탬프 |

## 3. 결과

### 3.1 폰트 종류 (id 1–12)

![타이포그래피 프로브 1](../../docs/ovis_probe_sheet1_20260720.jpg)

### 3.2 표현 방식 · 배치 · 신규 장면 (id 13–24)

![타이포그래피 프로브 2](../../docs/ovis_probe_sheet2_20260720.jpg)

## 4. 관찰

두 가지를 따로 봐야 한다 — **철자가 맞았는가**와 **요청한 서체로 나왔는가**.

| 항목 | 결과 |
| --- | --- |
| **철자** | 24개 중 **12개 정확** (HARBOR · Rosemary · VELVET · CARGO 47 · Splash · Andante · HOLLOW · SUNSET FM · FIELD/NOTES · MIDNIGHT EXPRESS 등) |
| 철자 오류 유형 | **글자 누락**이 대부분 — `ZENIH`(T), `IRONSDE`(I), `DEPRTURE`(A), `ATELER`(I), `ORHARD`(C), `MIDNGHT`(I) |
| 심한 붕괴 | `thank you`→`tihark`, `SYS_READY`→`SYS =ADU`, `LEVEL UP`→`LEVEL EVEP UPP`, `WOODLAND`→`WODLAD`, `BrightSide`→`BPigSHide`, `EST. 1968`→`ESTC 1968` |
| **서체** | **거의 전부 실패.** serif를 요청한 HARBOR가 산세리프로, calligraphy를 요청한 VELVET이 평범한 세리프로, italic을 요청한 Andante가 직립으로 나왔다. blackletter · condensed · pixel · stencil도 반영되지 않았다 |
| 서체 부분 성공 | script(Rosemary) · handwritten(글자는 틀렸으나 필기체 질감) · bubble(Splash)만 어렴풋이 반영 |
| 표현 방식 | emboss(ATELIER) · embroidery(SUMMIT) · 네온(MIDNIGHT EXPRESS) · 금박(EST. 1968)은 **재질 표현이 살아 있다**. chrome · gradient · outline은 약함 |
| 배치 | 2행 배치(FIELD / NOTES)는 성공. 숫자(CARGO 47 · PLATFORM 9)도 형태는 나오나 `9`가 두 번 찍혔다 |

## 5. 정리 — 2라운드 데이터에 주는 함의

- **서체 제어는 능력이 아니라 학습의 부재다.** 캡션에 서체 단어가 0%이므로 모델은 그 축을 조건으로 쓸 방법을 배운 적이 없다. 요청을 무시하고 학습셋에서 가장 흔한 굵은 산세리프로 되돌아간다. **캡션에 서체 슬롯을 넣는 것만으로 열릴 여지가 있는 축**이다.
- **재질 표현은 이미 된다.** emboss·embroidery·네온·금박은 요청대로 나왔다. 이것들은 서체가 아니라 **장면·재질**의 속성이고, 학습셋 9종 장면에 유사한 것이 있었기 때문으로 보인다. 서체와 재질을 구분해서 계획해야 한다.
- **철자는 여전히 절반이다.** 24개 중 12개만 정확하고, 오류는 대부분 **글자 하나 누락**이다. [vs_prx.md](vs_prx.md) 5.3절에서 확인한 "획득 후 흔들림"과 같은 성격이며, 학습셋 텍스트가 106개 단어에 갇혀 있는 것이 원인일 가능성이 크다. **새 단어에서 특히 약하다.**
- **텍스트 어휘를 넓혀야 한다.** `PIERROT` 하나가 38%를 차지하는 분포로는 임의의 단어를 쓰는 능력이 자라기 어렵다. 2라운드에서는 **단어 수를 수천 종으로 늘리고 특정 단어 편중을 없애야** 한다.
- **한계** — 프롬프트당 1장·seed 1개다. 철자는 seed에 민감하므로 판정을 확정으로 읽으면 안 된다.

## 6. 관련 문서

- [person_coverage_probe.md](person_coverage_probe.md) — 같은 형식의 인물 학습셋 진단
- [round1_experiment_report.md](round1_experiment_report.md) — 1차 실험 결산 (3절 학습셋 문제)
- [vs_prx.md](vs_prx.md) — 외부 모델 비교 (5.3절 텍스트 렌더링 획득 곡선)
- [ovis_probe_prompts.json](../ovis_probe_prompts.json) — 프롬프트 24개 전문 + 학습셋 분포 수치
