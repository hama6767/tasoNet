#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset_audio_prep.py で生成された segments.csv を読み込み、
Whisper を使って text を付与した segments_with_text.csv を出力するスクリプト。

使い方（例）:
    python transcribe_segments.py data/streamer01 \
        --model large-v3 --device auto --speaker streamer01
"""

import argparse
import csv
import contextlib
import wave
from pathlib import Path
from typing import Optional, Callable, List, Dict

import torch
import whisper


def get_wav_duration_sec(path: Path) -> float:
    """mono wav の長さを秒で返す（簡易版）"""
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate == 0:
        return 0.0
    return frames / float(rate)


def transcribe_segments(
    dataset_dir: Path,
    segments_csv: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    model_name: str = "large-v3",
    device: str = "auto",
    language: str = "ja",
    speaker: str = "spk",
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    segments.csv を読み込み、Whisperで文字起こしして out_csv に保存する。

    Args:
        dataset_dir: dataset_audio_prep.py の出力ディレクトリ
        segments_csv: 読み込むCSVパス（未指定なら dataset_dir/segments.csv）
        out_csv: 出力CSVパス（未指定なら dataset_dir/segments_with_text.csv）
        model_name: Whisperモデル名（tiny/base/small/medium/large-v3 等）
        device: "auto" / "cuda" / "cpu"
        language: 言語コード（日本語なら "ja"）
        speaker: speaker列のデフォルト値
        overwrite: True の場合、既に text が入っている行も上書き
        progress_callback: ログ文字列を受け取るコールバック（GUI用）
    """
    dataset_dir = dataset_dir.resolve()
    if segments_csv is None:
        segments_csv = dataset_dir / "segments.csv"
    if out_csv is None:
        out_csv = dataset_dir / "segments_with_text.csv"

    if not segments_csv.is_file():
        raise FileNotFoundError(f"segments_csv が見つかりません: {segments_csv}")

    # CSV読み込み
    with segments_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
        base_fieldnames = reader.fieldnames or []

    # 出力フィールド名を決定（共通フォーマット: speaker, text, lang, duration_sec）
    fieldnames = list(base_fieldnames)
    for col in ["speaker", "text", "lang", "duration_sec"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # デバイス選択
    if device == "auto":
        device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_resolved = device

    def log(msg: str):
        if progress_callback is not None:
            progress_callback(msg)
        else:
            print(msg)

    log(f"Loading Whisper model '{model_name}' on {device_resolved}...")
    model = whisper.load_model(model_name, device=device_resolved)
    fp16 = device_resolved != "cpu"

    total = len(rows)
    use_tqdm = progress_callback is None

    if use_tqdm:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(rows, start=1), total=total, desc="Transcribing")
        except ImportError:
            iterator = enumerate(rows, start=1)
    else:
        iterator = enumerate(rows, start=1)

    new_rows: List[Dict[str, str]] = []

    for idx, row in iterator:
        utt_id = row.get("utt_id", f"utt{idx:04d}")
        relpath = row.get("relpath")
        if not relpath:
            log(f"[WARN] row {idx} に relpath がありません。スキップします。")
            continue

        wav_path = (dataset_dir / relpath).resolve()
        if not wav_path.is_file():
            log(f"[WARN] wav が見つかりません: {wav_path} (utt_id={utt_id})")
            continue

        # speaker デフォルト
        if not row.get("speaker"):
            row["speaker"] = speaker

        # duration_sec が無ければ計算
        if not row.get("duration_sec"):
            dur = get_wav_duration_sec(wav_path)
            row["duration_sec"] = f"{dur:.3f}"

        # 既存textの扱い
        if (not overwrite) and row.get("text"):
            # lang が未設定なら補完
            if not row.get("lang"):
                row["lang"] = language
            new_rows.append(row)
            continue

        log(f"[{idx}/{total}] Transcribing {utt_id} ...")

        # Whisperによる文字起こし
        # パスを渡すだけで ffmpeg 経由で読み込んでくれる
        result = model.transcribe(
            str(wav_path),
            language=language,
            task="transcribe",
            verbose=False,
            fp16=fp16,
        )
        text = result.get("text", "").strip()
        row["text"] = text
        row["lang"] = language

        new_rows.append(row)

    # 出力
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in new_rows:
            # 欠けている列があっても落ちないように
            for col in fieldnames:
                r.setdefault(col, "")
            writer.writerow(r)

    log(f"Done. {len(new_rows)} rows written to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="segments.csv を Whisper で文字起こしして segments_with_text.csv を生成する"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="dataset_audio_prep.py で作成した出力ディレクトリ",
    )
    parser.add_argument(
        "--segments_csv",
        type=Path,
        default=None,
        help="入力CSV (未指定なら <dataset_dir>/segments.csv)",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="出力CSV (未指定なら <dataset_dir>/segments_with_text.csv)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisperモデル名（例: tiny, base, small, medium, large-v3）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="使用するデバイス（auto: CUDAがあればCUDA, なければCPU）",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="文字起こし言語コード（日本語なら ja）",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="spk",
        help="speaker列のデフォルト値（マルチ話者でなければ1種類でOK）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既に text が入っている行も上書きする",
    )

    args = parser.parse_args()

    transcribe_segments(
        dataset_dir=args.dataset_dir,
        segments_csv=args.segments_csv,
        out_csv=args.out_csv,
        model_name=args.model,
        device=args.device,
        language=args.language,
        speaker=args.speaker,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
