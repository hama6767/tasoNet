#!/usr/bin/env bash
set -eu

IMAGE="gpt-sovits-train"       # あなたの自作イメージ名
GPU="0"
HOST_WS="/home/hama6767/DA/tasoNet"         # ホスト側の /host_ws の実パスに合わせて

TEXT="${1:-よろしくね。}"
PROMPT_TEXT="ちょっと、おブサお魚の中で、大変有名なお方ではないでしょうか"
REF_WAV="${HOST_WS}/tsukuyomi.wav"
OUT_WAV="${HOST_WS}/out/test_001.wav"

docker run --rm \
  --gpus "device=${GPU}" \
  -v "${HOST_WS}:${HOST_WS}" \
  -v "$(pwd):/workspace" \
  -e TRANSFORMERS_NO_TORCHVISION=1 \
  "${IMAGE}" \
  python /workspace/GPT-SoVITS/infer_cli.py \
    --config /workspace/GPT-SoVITS/GPT_SoVITS/configs/s2.json \
    --bert-path /workspace/GPT-SoVITS/GPT-SoVITS/pretrained_models/bert-base-multilingual-cased \
    --gpt-ckpt /home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/s1v3.ckpt \
    --s2-ckpt /home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/s2G488k.pth \
    --ref-wav "${REF_WAV}" \
    --prompt-text "${PROMPT_TEXT}" \
    --text "${TEXT}" \
    --out-wav "${OUT_WAV}" \
    --no-half