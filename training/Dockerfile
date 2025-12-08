# ファイル名例: Dockerfile.gpt_sovits_train
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# 必要パッケージ
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- GPT-SoVITS 本体をイメージ内にコピー ----
# カレントディレクトリに GPT-SoVITS/ がある前提
COPY GPT-SoVITS /workspace/GPT-SoVITS

# ---- 学習パイプライン（後で書く gpt_sovits_train_inner.py）をコピー ----
COPY gpt_sovits_train_inner.py /workspace/gpt_sovits_train_inner.py

WORKDIR /workspace/GPT-SoVITS

# ---- Python 依存をインストール ----
# 本家の requirements と extra-req を利用
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -f extra-req.txt ]; then pip install --no-cache-dir -r extra-req.txt; fi

# 既定の作業ディレクトリは GPT-SoVITS
WORKDIR /workspace/GPT-SoVITS

CMD ["/bin/bash"]
