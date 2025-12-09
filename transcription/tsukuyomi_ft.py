from pathlib import Path

# ★ここを自分の環境に合わせて書き換えてください
WORKSPACE_ROOT = Path("/home/hama6767/DA/tasoNet/dataset/tsukuyomi")
SCRIPT_PATH = WORKSPACE_ROOT / "scripts.txt"
WAV_DIR = WORKSPACE_ROOT / "wav"        # VOICEACTRESS100_001.wav などが入っているディレクトリ
LIST_PATH = WORKSPACE_ROOT / "gpt_sovits_tsukuyomi.list"

SPEAKER_NAME = "tsukuyomi"
LANG = "ja"

lines = []

with SCRIPT_PATH.open("r", encoding="utf-8") as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        # VOICEACTRESS100_001:テキスト...
        try:
            utt_id, text = raw.split(":", 1)
        except ValueError:
            print("skip malformed line:", raw)
            continue

        utt_id = utt_id.strip()
        text = text.strip()
        if not text:
            continue

        wav_path = WAV_DIR / f"{utt_id}.wav"
        if not wav_path.is_file():
            print("!! wav not found:", wav_path)
            continue

        # 絶対パスにしておく
        wav_abs = wav_path.resolve()
        line = f"{wav_abs}|{SPEAKER_NAME}|{LANG}|{text}"
        lines.append(line)

with LIST_PATH.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("wrote", LIST_PATH, "lines:", len(lines))
