#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segments_with_text.csv を GPT-SoVITS 用の .list に変換するスクリプト。

入力:
    - 1つ以上の segments_with_text.csv
      (列: utt_id, relpath, speaker, text, lang, duration_sec など)

出力:
    - wav_path|spk_name|language|text
      形式の .list ファイル

使い方例:

    # 単純に1つの CSV から list を作る
    python make_gpt_sovits_list.py \
        data/streamer/segments_with_text.csv \
        --out-list gpt_sovits_streamer.list \
        --speaker streamer01

    # JSUT + JVS + 配信者をまとめて1つの list に
    python make_gpt_sovits_list.py \
        data/jsut/segments_with_text.csv \
        data/jvs/segments_with_text.csv \
        data/streamer/segments_with_text.csv \
        --out-list gpt_sovits_all.list \
        --min-duration 0.5 \
        --max-duration 15.0 \
        --lang-filter ja

    # train / val に分割して保存
    python make_gpt_sovits_list.py \
        data/streamer/segments_with_text.csv \
        --out-list gpt_sovits_streamer_train.list \
        --val-list gpt_sovits_streamer_val.list \
        --val-ratio 0.02 \
        --speaker streamer01
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional


def load_rows(csv_paths: List[Path]) -> List[Dict[str, str]]:
    """
    複数の segments_with_text.csv を読み込み、
    各行に _base_dir (CSVの親ディレクトリ) を付与して返す。
    """
    all_rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV が見つかりません: {csv_path}")

        base_dir = csv_path.parent

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for r in rows:
            r["_base_dir"] = str(base_dir)
        all_rows.extend(rows)

    return all_rows


def parse_duration(row: Dict[str, str]) -> Optional[float]:
    """
    duration_sec 列があればそれを、なければ end_sec - start_sec から計算。
    どちらも無ければ None。
    """
    if row.get("duration_sec"):
        try:
            return float(row["duration_sec"])
        except ValueError:
            pass

    if row.get("start_sec") and row.get("end_sec"):
        try:
            return float(row["end_sec"]) - float(row["start_sec"])
        except ValueError:
            pass

    return None


def filter_rows(
    rows: List[Dict[str, str]],
    min_duration: float,
    max_duration: float,
    speakers: Optional[List[str]] = None,
    lang_filter: Optional[str] = None,
    require_text: bool = True,
) -> List[Dict[str, str]]:
    """
    時間・話者・言語などの条件で rows をフィルタリングする。
    """
    result: List[Dict[str, str]] = []

    speakers_set = set(speakers) if speakers else None

    for r in rows:
        # duration
        dur = parse_duration(r)
        if dur is None:
            # durationが不明な行はスキップ
            continue
        if dur < min_duration or dur > max_duration:
            continue

        # text 必須
        text = (r.get("text") or "").strip()
        if require_text and not text:
            continue

        # speaker フィルタ
        spk = (r.get("speaker") or "").strip()
        if speakers_set is not None and spk not in speakers_set:
            continue

        # 言語フィルタ
        lang = (r.get("lang") or "").strip()
        if lang_filter is not None:
            # lang が空なら不採用にしておく（必要ならここを柔らかくしても良い）
            if lang != lang_filter:
                continue

        result.append(r)

    return result


def row_to_line_gpt_sovits(r: Dict[str, str]) -> str:
    """
    1行の dict から GPT-SoVITS 形式:
      wav_path|spk_name|language|text
    の文字列を生成。
    """
    base_dir = Path(r["_base_dir"])
    relpath = (r.get("relpath") or "").strip()
    wav_path = (base_dir / relpath).resolve()

    # speaker / lang / text
    spk = (r.get("speaker") or "spk").strip()
    lang = (r.get("lang") or "ja").strip()
    text = (r.get("text") or "").replace("\n", " ").strip()

    return f"{wav_path}|{spk}|{lang}|{text}"


def write_list(path: Path, lines: List[str]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="segments_with_text.csv から GPT-SoVITS 用 .list を生成する"
    )
    ap.add_argument(
        "segments_csv",
        type=Path,
        nargs="+",
        help="入力 segments_with_text.csv (複数指定可)",
    )
    ap.add_argument(
        "--out-list",
        type=Path,
        required=True,
        help="出力する train 用 .list のパス",
    )
    ap.add_argument(
        "--val-list",
        type=Path,
        default=None,
        help="validation 用 .list のパス (val-ratio > 0 のときのみ有効)",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="train/val の分割比 (0 なら全て train に入れる)",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=0.3,
        help="最低発話長 (秒)",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="最大発話長 (秒)",
    )
    ap.add_argument(
        "--speaker",
        type=str,
        action="append",
        default=None,
        help="特定の speaker のみを使いたい場合に指定 (複数可, 例: --speaker jsut --speaker streamer01)",
    )
    ap.add_argument(
        "--lang-filter",
        type=str,
        default=None,
        help="特定の lang のみを使う場合に指定 (例: ja)",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="行をシャッフルしてから train/val 分割する",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="シャッフルに使う乱数シード",
    )

    args = ap.parse_args()

    # ---- CSVを読み込み ----
    rows = load_rows(args.segments_csv)
    print(f"[INFO] CSV 合計行数: {len(rows)}")

    # ---- フィルタリング ----
    rows = filter_rows(
        rows,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        speakers=args.speaker,
        lang_filter=args.lang_filter,
        require_text=True,
    )
    print(f"[INFO] フィルタ後行数: {len(rows)}")

    if not rows:
        print("[WARN] 有効な行がありません。条件を緩めて再実行してください。")
        return

    # ---- シャッフル ----
    if args.shuffle or args.val_ratio > 0:
        rnd = random.Random(args.seed)
        rnd.shuffle(rows)

    # ---- train / val 分割 ----
    if args.val_ratio > 0:
        n_total = len(rows)
        n_val = max(1, int(n_total * args.val_ratio))
        val_rows = rows[:n_val]
        train_rows = rows[n_val:]
    else:
        train_rows = rows
        val_rows = []

    # ---- 行を GPT-SoVITS 形式に変換 ----
    train_lines = [row_to_line_gpt_sovits(r) for r in train_rows]
    val_lines = [row_to_line_gpt_sovits(r) for r in val_rows]

    # ---- 保存 ----
    write_list(args.out_list, train_lines)
    print(f"[INFO] train: {len(train_lines)} 行 -> {args.out_list}")

    if args.val_ratio > 0 and val_lines:
        if args.val_list is None:
            # out-list のファイル名に _val を足したものをデフォルトにする
            default_val = args.out_list.with_name(
                args.out_list.stem + "_val" + args.out_list.suffix
            )
            val_path = default_val
        else:
            val_path = args.val_list

        write_list(val_path, val_lines)
        print(f"[INFO]  val: {len(val_lines)} 行 -> {val_path}")


if __name__ == "__main__":
    main()
