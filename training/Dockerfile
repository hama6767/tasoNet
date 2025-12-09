FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV TRANSFORMERS_NO_TORCHVISION=1

COPY GPT-SoVITS /workspace/GPT-SoVITS
COPY gpt_sovits_train_inner.py /workspace/gpt_sovits_train_inner.py

WORKDIR /workspace/GPT-SoVITS

# 依存インストール
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -f extra-req.txt ]; then pip install --no-cache-dir -r extra-req.txt; fi && \
    # ★ 壊れた torchvision / flash-attn 系を削除（存在しなくてもOK）
    pip uninstall -y torchvision flash-attn flash_attn flash_attn-2 flash_attn_2 || true

RUN pip install --no-cache-dir "transformers>=4.52.1" "peft==0.18.0"

# ★ BERT モデルを事前ダウンロード
RUN python - << 'EOF'
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "bert-base-multilingual-cased"
save_dir = Path("/workspace/GPT-SoVITS/GPT-SoVITS/pretrained_models/bert-base-multilingual-cased")
save_dir.mkdir(parents=True, exist_ok=True)

print("Downloading", model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("Saved BERT model to", save_dir)
EOF

COPY infer_cli.py /workspace/GPT-SoVITS/infer_cli.py

WORKDIR /workspace/GPT-SoVITS

CMD ["/bin/bash"]
