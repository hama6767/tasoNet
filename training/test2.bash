#!/usr/bin/env bash
set -eu

IMAGE="gpt-sovits-train"       # あなたの自作イメージ名
GPU="0"
HOST_WS="/home/hama6767/DA/tasoNet"         # ホスト側の /host_ws の実パスに合わせて

TEXT="${1:-あまねかなただよ、こんかなた。よろしくね。}"
PROMPT_TEXT="また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
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
    --gpt-ckpt /home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
    --s2-ckpt /home/hama6767/DA/tasoNet/tsukuyomi.pth \
    --ref-wav "${REF_WAV}" \
    --prompt-text "${PROMPT_TEXT}" \
    --text "${TEXT}" \
    --out-wav "${OUT_WAV}" \
    --no-half