#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
コンテナ内で使う GPT-SoVITS 推論用 CLI

例:

python /workspace/GPT-SoVITS/infer_cli.py \
  --config /workspace/GPT-SoVITS/GPT_SoVITS/configs/s2.json \
  --s2-ckpt /home/hama6767/DA/tasoNet/SoVITS_weights_v2/exp_streamer_best.pth \
  --ref-wav /host_ws/ref_speaker.wav \
  --text "こんにちは、テストです。" \
  --out-wav /host_ws/out/test_001.wav
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch

# あなたの環境に合わせたデフォルト
DEFAULT_BERT_PATH = "/workspace/GPT-SoVITS/GPT-SoVITS/pretrained_models/bert-base-multilingual-cased"
DEFAULT_CNHUBERT_PATH = "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"

TTS: Any = None
I18N: Any = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="S2 用 config JSON (s2.json)")
    ap.add_argument("--s2-ckpt", required=True, dest="s2_ckpt",
                    help="SoVITS 学習済み ckpt (.pth)")
    ap.add_argument("--ref-wav", required=True, dest="ref_wav",
                    help="話者条件リファレンス wav")
    ap.add_argument("--text", required=True, help="合成したいテキスト")
    ap.add_argument("--out-wav", required=True, dest="out_wav",
                    help="出力 wav パス")

    ap.add_argument("--gpt-ckpt", required=False,
                    help="任意の GPT(Stage1) ckpt。未指定ならデフォルトを使用。")
    ap.add_argument("--bert-path", default=DEFAULT_BERT_PATH,
                    help="BERT モデルディレクトリ")
    ap.add_argument("--cnhubert-path", default=DEFAULT_CNHUBERT_PATH,
                    help="CN-HuBERT base ディレクトリ")
    ap.add_argument("--no-half", action="store_true",
                    help="fp16 を使わず fp32 推論する場合に指定")

    return ap.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for name in ("config", "s2_ckpt", "ref_wav"):
        p = Path(getattr(args, name))
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    out_dir = Path(args.out_wav).parent
    out_dir.mkdir(parents=True, exist_ok=True)


def setup_env(args: argparse.Namespace) -> None:
    os.environ["bert_path"] = args.bert_path
    os.environ["cnhubert_base_path"] = args.cnhubert_path
    os.environ["is_half"] = "False" if args.no_half else "True"
    if args.gpt_ckpt:
        os.environ["gpt_path"] = args.gpt_ckpt


def import_tts_modules() -> None:
    global TTS, I18N

    repo_root = Path("/workspace/GPT-SoVITS").resolve()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from GPT_SoVITS import inference_webui as tts_mod
    from tools.i18n.i18n import I18nAuto

    TTS = tts_mod
    I18N = I18nAuto()


def load_s2_model(config_path: str, ckpt_path: str, device: torch.device) -> Tuple[Any, Any]:
    assert TTS is not None

    # config も一応読む（サンプリングレート確認用）
    try:
        from GPT_SoVITS import utils
        hps_file = utils.get_hparams_from_file(config_path)
        print(f"[info] config sr = {hps_file.data.sampling_rate}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed to read config: {e}", file=sys.stderr)

    # ckpt 内の config でモデルロード
    gen = TTS.change_sovits_weights(ckpt_path)
    try:
        next(gen)
    except StopIteration:
        pass

    model = getattr(TTS, "vq_model", None)
    hps = getattr(TTS, "hps", None)
    if model is None or hps is None:
        raise RuntimeError("failed to load SoVITS model via change_sovits_weights")

    print("[info] SoVITS loaded.")
    return model, hps


def load_gpt_model(ckpt_path: str | None) -> None:
    assert TTS is not None
    if ckpt_path:
        print(f"[info] loading GPT weights from {ckpt_path}")
        TTS.change_gpt_weights(ckpt_path)
    else:
        print("[info] using default GPT weights.")


def extract_features(ref_wav: str, text: str) -> Dict[str, Any]:
    assert I18N is not None
    # 「多语种混合」を i18n して WebUI と同じキーを使う
    lang = I18N("多语种混合")
    return {
        "ref_wav": ref_wav,
        "prompt_text": "",
        "prompt_lang": lang,
        "text": text,
        "text_lang": lang,
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
    }


def synthesize(model: Any, hps: Any, feat: Dict[str, Any]) -> Tuple[int, np.ndarray]:
    assert TTS is not None

    gen = TTS.get_tts_wav(
        ref_wav_path=feat["ref_wav"],
        prompt_text=feat["prompt_text"],
        prompt_language=feat["prompt_lang"],
        text=feat["text"],
        text_language=feat["text_lang"],
        top_k=feat["top_k"],
        top_p=feat["top_p"],
        temperature=feat["temperature"],
    )

    sr = None
    audio = None
    for sr_chunk, audio_chunk in gen:
        sr = sr_chunk
        audio = audio_chunk

    if sr is None or audio is None:
        raise RuntimeError("get_tts_wav did not yield anything")

    if audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    return int(sr), audio


def save_wav(sr: int, audio: np.ndarray, out_path: str) -> None:
    sf.write(out_path, audio, sr)
    print(f"[info] saved: {out_path} (sr={sr}, dur={len(audio)/sr:.2f}s)")


def main() -> None:
    args = parse_args()
    try:
        validate_paths(args)
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    setup_env(args)
    import_tts_modules()
    s2_model, hps = load_s2_model(args.config, args.s2_ckpt, device)
    load_gpt_model(args.gpt_ckpt)

    feat = extract_features(args.ref_wav, args.text)
    sr, audio = synthesize(s2_model, hps, feat)
    save_wav(sr, audio, args.out_wav)


if __name__ == "__main__":
    main()
