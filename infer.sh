#!/bin/bash
# PIERROT 단일 프롬프트 추론 wrapper.
#
# 사용 예:
#   CKPT=checkpoints/0.8b_base/model.pt \
#   PROMPT="a red apple on a wooden table" \
#   bash PIERROT_INFER/infer.sh
#
# env 로 override 가능: CKPT(필수) / PROMPT / OUTPUT / GPU / SEED / STEPS / CFG / PYTHON_BIN
# 경로는 스크립트 위치 기반 자동 도출.
set -euo pipefail

INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = .../PIERROT_INFER
STUDY_ROOT="$(cd "${INFER_DIR}/.." && pwd)"                 # PIERROT_INFER import 가능 위치

CKPT="${CKPT:?CKPT 환경변수 필수 (예: CKPT=checkpoints/0.8b_base/model.pt)}"
PROMPT="${PROMPT:-a red apple on a wooden table}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
STEPS="${STEPS:-28}"
CFG="${CFG:-4.0}"
DATE="$(date +%Y%m%d)"
OUTPUT="${OUTPUT:-${INFER_DIR}/results/pierrot_${DATE}.png}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "${OUTPUT}")"
cd "${STUDY_ROOT}"

echo "[INFO] CKPT   = ${CKPT}"
echo "[INFO] PROMPT = ${PROMPT}"
echo "[INFO] OUTPUT = ${OUTPUT}"
echo "[INFO] GPU=${GPU}  SEED=${SEED}  STEPS=${STEPS}  CFG=${CFG}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m PIERROT_INFER.sample \
    --ckpt   "${CKPT}" \
    --prompt "${PROMPT}" \
    --output "${OUTPUT}" \
    --seed   "${SEED}" \
    --steps  "${STEPS}" \
    --cfg    "${CFG}"

echo "[DONE] saved: ${OUTPUT}"
