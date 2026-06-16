#!/bin/bash
# PIERROT 77 prompt 일괄 추론 wrapper (11 standard + 5 deepfashion + 1 dog_bench + 5 anyword + 5 hybrid + 6 FC test + 3 추가 + 14 다양성 + 6 악기 + 5 PFM + 5 aw2 + 5 aw1 + 6 awreal, 2026-06-10).
#
# 환경변수로 ckpt step / GPU / seed / phase / 날짜 변경 가능.
# 메모리 룰 — 새 추론 PNG 에 항상 _YYYYMMDD suffix 자동 적용 (이전 PNG 보존).
#
# ── 사용 예 ──
#
# 1. 기본 (phase2 + GPU 7 + 오늘 날짜):
#      CKPT_STEP=01480000 bash PIERROT/infer.sh
#
# 2. 다른 GPU 사용:
#      CKPT_STEP=01480000 GPU=3 bash PIERROT/infer.sh
#
# 3. phase1 추론:
#      CKPT_STEP=00120000 PHASE=phase1 bash PIERROT/infer.sh
#
# 4. 모든 변수 한 번에 변경:
#      CKPT_STEP=01480000 GPU=0 SEED=7 DATE=20260601 PHASE=phase2 \
#          bash PIERROT/infer.sh
#
# ── 출력 파일명 규칙 (메모리 룰) ──
#   pierrot_step_<CKPT_STEP>_ema_<tag>_<DATE>.png   (cat 만 tag 없음)
#   → PIERROT/results/ 안 저장.   이전 PNG 는 _YYYYMMDD 다른 suffix 라 자동 보존.

set -euo pipefail

# ── 환경변수 / 기본값 ──
CKPT_STEP="${CKPT_STEP:?CKPT_STEP 환경변수 필수 (예: CKPT_STEP=01480000)}"
GPU="${GPU:-7}"
SEED="${SEED:-42}"
PHASE="${PHASE:-phase2}"
DATE="${DATE:-$(date +%Y%m%d)}"

# ── 경로 (스크립트 위치 기반 자동 도출, env 로 override 가능) ──
INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = .../PIERROT_INFER
STUDY_ROOT="$(cd "${INFER_DIR}/.." && pwd)"                 # = PIERROT_INFER 의 상위
PIERROT_ROOT="${PIERROT_ROOT:-${STUDY_ROOT}/PIERROT}"      # ckpt 보관 위치 (env override 가능)
CKPT_PATH="${CKPT_PATH:-${PIERROT_ROOT}/checkpoints/${PHASE}/step_${CKPT_STEP}/ema.pt}"
OUT_DIR="${OUT_DIR:-${INFER_DIR}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── 사전 검증 ──
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[ERROR] ckpt 파일 없음: ${CKPT_PATH}" >&2
    echo "        사용 가능한 step:" >&2
    ls "${PIERROT_ROOT}/checkpoints/${PHASE}/" 2>/dev/null | sort -V | tail -10 >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ── 16 prompt 정의 (메모리 룰 — feedback_pierrot_inference_5_prompts.md, 2026-06-01 deepfashion 5 추가) ──
#    fashion_* 5 = DeepFashion-MultiModal 학습 신호 검증용 (인간 풀바디 anatomy, 의류 prior, multi-subject 의류 구분)
declare -A PROMPTS=(
    [cat]="a cat sitting on a wooden bench"
    [apple]="a red apple on a wooden table"
    [woman]="a portrait of a woman wearing a blue jacket"
    [city]="a busy city street at night with neon signs"
    [cabin]="a cozy cabin interior with warm lighting, detailed textures"
    [dog]="a golden retriever puppy playing with a ball in a green park"
    [food]="a freshly baked pizza with melted cheese and pepperoni on a wooden cutting board"
    [car]="a vintage red sports car parked on a coastal road at sunset"
    [landscape]="a snowy mountain landscape with pine trees and a frozen lake at dawn"
    [couple]="a young man and woman couple smiling and holding hands on a beach"
    [neon]='a vibrant neon sign showing the word "PIERROT" in glowing pink and blue letters at night, mounted on a dark brick wall, photorealistic, cinematic lighting'
    # ── deepfashion 5 (2026-06-01) — 풀바디 anatomy / 의류 차별 / multi-subject 검증 ──
    [fashion_dress]="a fashion model wearing an elegant black evening dress, full body shot, studio lighting, photorealistic"
    [fashion_suit]="a man wearing a tailored navy blue business suit with white shirt, full body fashion shot, photorealistic"
    [fashion_streetwear]="a young woman wearing a denim jacket, white t-shirt, ripped jeans, and white sneakers, full body street style photography"
    [fashion_floral]="a woman wearing a floral summer sundress and sandals, full body outdoor fashion shot, soft natural light"
    [fashion_couple]="a young couple, man in a gray suit and woman in a red cocktail dress, full body fashion editorial shot, studio lighting"
    # ── 추가 (2026-06-02) — dog 의 action+multi-object 약점 분리 검증 ──
    [dog_bench]="a dog sitting on a wooden bench"
    # ── anyword 5 (2026-06-06) — anyword_laion caption format 매칭 char-level 글자 학습 검증 ──
    [anyword_poster]='a vintage movie poster on a brick wall. The visible text reads "CINEMA", "1985".'
    [anyword_cafe]='a coffee shop sign hanging on a wooden door. The visible text reads "OPEN", "COFFEE".'
    [anyword_pierrot]='a glossy magazine cover with a portrait. The visible text reads "PIERROT".'
    [anyword_sticker]='a yellow sticker on a glass bottle. The visible text reads "FRESH", "JUICE".'
    [anyword_podcast]='a podcast cover with a microphone illustration. The visible text reads "STORIES", "PODCAST".'
    # ── hybrid 5 (2026-06-06) — anyword format + photorealistic 키워드 (quality 효과 측정) ──
    [hybrid_poster]='a vintage movie poster on a brick wall, photorealistic, cinematic lighting, detailed textures. The visible text reads "CINEMA", "1985".'
    [hybrid_cafe]='a coffee shop sign hanging on a wooden door, photorealistic, soft natural light, detailed. The visible text reads "OPEN", "COFFEE".'
    [hybrid_pierrot]='a glossy magazine cover with a portrait, photorealistic, professional studio lighting, high detail. The visible text reads "PIERROT".'
    [hybrid_sticker]='a yellow sticker on a glass bottle, photorealistic, sharp detail, natural light. The visible text reads "FRESH", "JUICE".'
    [hybrid_podcast]='a podcast cover with a microphone illustration, photorealistic, professional design, detailed. The visible text reads "STORIES", "PODCAST".'
    # ── Fashion Couple Test Set F0~F5 (2026-06-07) — multi-subject gender + attribute binding 진단 ──
    #     F0: 자연 프롬프트 일반화 / F1: one/one 수량 / F2: heterosexual / F3: 문장 분리 binding
    #     F4: slot-first / F5: position (left/right) binding
    [fc_F0]='a young couple, man in a gray suit and woman in a red cocktail dress, full body fashion editorial shot, studio lighting'
    [fc_F1]='one young man in a gray suit and one young woman in a red cocktail dress, full body fashion editorial shot, studio lighting'
    [fc_F2]='a heterosexual fashion couple, one man wearing a gray suit and one woman wearing a red cocktail dress, full body studio fashion editorial'
    [fc_F3]='two people in a studio fashion editorial. the man is wearing a gray suit. the woman is wearing a red cocktail dress. full body shot'
    [fc_F4]='one man wearing a gray suit and one woman wearing a red cocktail dress, standing side by side, full body fashion editorial shot, studio lighting'
    [fc_F5]='two people standing side by side in a studio. on the left, one man wearing a gray suit. on the right, one woman wearing a red cocktail dress. full body fashion editorial shot'
    # ── 추가 3 prompt (2026-06-08) — 일반 quality 검증 ──
    #     grandfather_clock: 단일 object 디테일 / cat_anatomy: anatomy/structure / woman_realistic: realistic portrait
    [grandfather_clock]='a tall antique grandfather clock in a victorian hallway, warm lighting, photorealistic'
    [cat_anatomy]='an anatomy illustration of a cat, detailed scientific drawing, side view, labeled diagram style'
    [woman_realistic]='a portrait of a woman wearing a blue jacket, photorealistic, detailed skin and hair, sharp focus'
    # ── 다양성 14 prompt (2026-06-08) — 기존 카테고리와 안 겹치는 주제 확장 ──
    #     interior / object / nature / 인물(직업) / sci-fi / 건축 / art style 등 도메인 다양화 검증
    [bookshelf]='a wooden bookshelf filled with old leather-bound books in a dimly lit library, warm lamp light, photorealistic'
    [bowl_fruit]='a glass bowl filled with fresh strawberries, blueberries, and orange slices on a marble countertop, natural light, food photography'
    [forest_path]='a misty forest path with tall pine trees and fallen leaves at autumn sunrise, soft sunlight rays piercing through mist, photorealistic'
    [lighthouse]='a white lighthouse on a rocky cliff overlooking a stormy ocean at dusk, dramatic clouds, photorealistic'
    [musician]='a young musician playing an acoustic guitar on a street corner at sunset, golden hour lighting, photorealistic'
    [dessert]='a slice of chocolate cake with raspberry sauce and mint leaves on a white ceramic plate, soft natural light, food photography'
    [vintage_camera]='a vintage film camera with leather strap on a wooden desk surrounded by old photographs, warm lighting, photorealistic'
    [flower_field]='a vast field of purple lavender flowers under a clear blue sky with white clouds in Provence, photorealistic'
    [chef]='a professional chef in white uniform plating a gourmet dish in a modern restaurant kitchen, action shot, photorealistic'
    [astronaut]='an astronaut in a white spacesuit standing on the surface of Mars with red rocky terrain and Earth visible in the sky, photorealistic'
    [owl_branch]='a great horned owl perched on a mossy tree branch in a misty forest at night with a full moon in the background, photorealistic'
    [hot_air_balloon]='colorful hot air balloons floating over Cappadocia at sunrise with a rocky landscape below, photorealistic'
    [train_station]='a vintage train arriving at an old European railway station with steam, passengers waiting on the platform, photorealistic'
    [watercolor_garden]='a watercolor painting of a Japanese zen garden with a stone lantern, cherry blossom tree, and koi pond, soft pastel colors'
    # ── 악기 6 (2026-06-08) — 사람 없이 악기 자체 디테일 + texture/lighting 검증 ──
    [acoustic_guitar]='a vintage acoustic guitar leaning against a wooden chair in a sunlit room, soft natural light through a window, sheet music scattered on the floor, photorealistic, detailed wood grain'
    [grand_piano]='a grand piano in an empty concert hall with warm stage lights, polished black wood surface reflecting the light, photorealistic, fine detail'
    [violin]='a vintage wooden violin with a bow resting on a music stand next to sheet music, warm natural window light, photorealistic, detailed wood grain'
    [drum_kit]='a vintage jazz drum kit on a small stage with warm spotlight, brass cymbals shining, photorealistic, detailed metal texture'
    [saxophone]='a golden brass saxophone leaning against a velvet chair in a dimly lit jazz lounge, photorealistic, soft warm light, detailed brass reflection'
    [cello]='a wooden cello with a bow placed beside a vintage music stand in a classical music room, soft window light, photorealistic, detailed wood grain'
    # ── PFM 5 (2026-06-09) — person_full_model VTON 식 outfit detail: single subject anatomy + ethnicity + outfit binding ──
    [pfm_F1]='a slim east asian man with short straight hair is wearing a regular navy blue cotton plain button up shirt with hip-length and a point collar, regular long beige cotton-linen plain trousers with a straight silhouette, and brown leather plain loafers with a round toe and a flat heel, with his shirt tucked in.'
    [pfm_F2]='an average-sized white woman with long wavy hair is wearing a regular black silk plain dress with knee-length and a v-line neckline, black synthetic heels with a pointed toe and a stiletto heel, and a small gold earring, with her hair styled neatly.'
    [pfm_F3]='a slim hispanic man with short curly hair is wearing a regular white cotton graphic t shirt with hip-length and a round neckline underneath a regular gray denim plain jacket with hip-length, regular long blue denim plain jeans, and white synthetic sneakers with a round toe and a flat heel.'
    [pfm_F4]='an average-sized black woman with shoulder-length straight hair is wearing a regular red cotton-linen plain blazer with hip-length and a point collar, a regular cream silk plain blouse with hip-length and a round neckline, slim long black plain trousers with a straight silhouette, and black leather plain pumps with a pointed toe and a kitten heel.'
    [pfm_F5]='an average-sized white man with shoulder-length wavy hair is wearing a regular dark green wool plain sweater with hip-length and a round neckline, slim long dark blue denim plain jeans, and brown leather plain boots with a round toe and a low heel, with his sweater untucked.'
    # ── aw2 5 (2026-06-09) — anyword_laion style 확장: 기존 anyword 5 외 다양 시나리오 (book/shirt/bottle/billboard/storefront) ──
    [aw2_book]='a hardcover book on a wooden table. The visible text reads "ATLAS", "EDITION".'
    [aw2_shirt]='a white cotton t-shirt on a hanger. The visible text reads "REBEL", "2030".'
    [aw2_bottle]='a glass beer bottle with a paper label. The visible text reads "CRAFT", "ALE".'
    [aw2_billboard]='a large outdoor billboard above a road. The visible text reads "EXIT", "5".'
    [aw2_storefront]='a small bakery storefront with a sign. The visible text reads "BREAD", "FRESH".'
    # ── aw1 5 (2026-06-09) — anyword_laion 1 단어 영역 (10.6% 분포): 다양 scene + 짧은/긴 단어 ──
    [aw1_logo]='a circular brand logo on a white background. The visible text reads "NOVA".'
    [aw1_neon]='a glowing neon sign in a dark alley at night. The visible text reads "OPEN".'
    [aw1_book]='a vintage leather-bound hardcover book on a wooden desk. The visible text reads "JOURNEY".'
    [aw1_tshirt]='a black cotton t-shirt on a white background. The visible text reads "WILD".'
    [aw1_billboard]='a large highway billboard above a road at dusk. The visible text reads "STOP".'
    # ── awreal 6 (2026-06-10) — 실제 AnyWord caption_original 스타일: 자연 scene + visible text clause ──
    [awreal_nr_bread]='a small bakery storefront with a bread sign above the door. The visible text reads "BREAD".'
    [awreal_nr_open]='a coffee shop door with an open sign hanging in the window. The visible text reads "OPEN".'
    [awreal_nr_nova]='a clean tech product box with the nova logo printed on the front. The visible text reads "NOVA".'
    [awreal_multi_bread]='a bakery display shelf with bread labels on several baskets. The visible text reads "BREAD".'
    [awreal_multi_coffee]='a cafe menu board with coffee labels on several menu sections. The visible text reads "COFFEE".'
    [awreal_multi_sale]='a shop window covered with sale stickers. The visible text reads "SALE".'
)

# ── 실행 ──
echo "[INFO] PHASE     = ${PHASE}"
echo "[INFO] CKPT_STEP = ${CKPT_STEP}"
echo "[INFO] CKPT_PATH = ${CKPT_PATH}"
echo "[INFO] GPU       = ${GPU}"
echo "[INFO] SEED      = ${SEED}"
echo "[INFO] DATE      = ${DATE} (output suffix)"
echo "[INFO] 출력 prefix = ${OUT_DIR}/pierrot_step_${CKPT_STEP}_ema_<tag>_${DATE}.png"
echo

cd "${STUDY_ROOT}"                                          # PIERROT_INFER 패키지 import 가능 위치

# 77 prompt sequential 추론
for tag in cat apple woman city cabin dog food car landscape couple neon \
           fashion_dress fashion_suit fashion_streetwear fashion_floral fashion_couple \
           dog_bench \
           anyword_poster anyword_cafe anyword_pierrot anyword_sticker anyword_podcast \
           hybrid_poster hybrid_cafe hybrid_pierrot hybrid_sticker hybrid_podcast \
           fc_F0 fc_F1 fc_F2 fc_F3 fc_F4 fc_F5 \
           grandfather_clock cat_anatomy woman_realistic \
           bookshelf bowl_fruit forest_path lighthouse musician dessert vintage_camera \
           flower_field chef astronaut owl_branch hot_air_balloon train_station watercolor_garden \
           acoustic_guitar grand_piano violin drum_kit saxophone cello \
           pfm_F1 pfm_F2 pfm_F3 pfm_F4 pfm_F5 \
           aw2_book aw2_shirt aw2_bottle aw2_billboard aw2_storefront \
           aw1_logo aw1_neon aw1_book aw1_tshirt aw1_billboard \
           awreal_nr_bread awreal_nr_open awreal_nr_nova \
           awreal_multi_bread awreal_multi_coffee awreal_multi_sale; do
    # cat 만 예외 — tag suffix 없음 (이전 호환 규칙)
    if [[ "${tag}" == "cat" ]]; then
        OUT="${OUT_DIR}/pierrot_step_${CKPT_STEP}_ema_${DATE}.png"
    else
        OUT="${OUT_DIR}/pierrot_step_${CKPT_STEP}_ema_${tag}_${DATE}.png"
    fi

    echo "===== [${tag}] $(date +%H:%M:%S) ====="
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m PIERROT_INFER.sample \
        --ckpt   "${CKPT_PATH}" \
        --prompt "${PROMPTS[$tag]}" \
        --output "${OUT}" \
        --seed   "${SEED}" \
        --steps  28
done

echo
echo "===== [DONE] $(date) ====="
echo "[INFO] 77 PNG 저장 완료 (suffix _${DATE}) — ${OUT_DIR}/"
