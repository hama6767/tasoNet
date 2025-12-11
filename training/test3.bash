#!/usr/bin/env bash
set -eu

IMAGE="gpt-sovits-train"       # あなたの自作イメージ名
GPU="0"
HOST_WS="/home/hama6767/DA/tasoNet"         # ホスト側の /host_ws の実パスに合わせて

TEXT="${1:-信じ続けた未来を裏切られた悲しみ覆すことが出来ない現実と夢を見続けた日々に挟まれ俺は己の弱さを憎み続けてきたそれでも伝え続けた言葉の欠片達ついには消えゆくその瞬間にあなたの心のどこかに少しでも残っていることを祈りそれが叶わぬ願いだと知りながら}"
REF_WAV="${HOST_WS}/taso3.wav"
OUT_WAV="${HOST_WS}/out/test_001.wav"
PROMPT_TEXT="ゲームはまだまだ上手くいかないこともあるけど最近はちょっと上手くなったような気がする一生懸命頑張っているから"

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
    --s2-ckpt /home/hama6767/DA/tasoNet/SoVITS_weights_v2/exp_streamer_e4_s452.pth \
    --ref-wav "${REF_WAV}" \
    --prompt-text "${PROMPT_TEXT}" \
    --text "${TEXT}" \
    --out-wav "${OUT_WAV}" \
    --no-half