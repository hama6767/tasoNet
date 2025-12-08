# Whisper + PyTorch 用 GPU イメージ
FROM nvcr.io/nvidia/pytorch:24.10-py3

# ===== ベースツール =====
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# VSCode devcontainer と同じ uid/gid のユーザを作る（任意）
RUN groupadd -g 1000 vscode && \
    useradd -m -u 1000 -g 1000 -s /bin/bash vscode

USER vscode
WORKDIR /workspace

# Python パッケージ（requirements.txt はプロジェクトルートからコピー）
# プロジェクト側に requirements.txt を置いておく
COPY --chown=vscode:vscode requirements.txt /workspace/requirements.txt
RUN pip3 install --user --no-cache-dir -r /workspace/requirements.txt

# pip --user で入れるので ~/.local/bin を PATH に追加
ENV PATH="/home/vscode/.local/bin:${PATH}"

CMD ["bash"]
