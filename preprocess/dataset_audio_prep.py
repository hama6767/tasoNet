#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mp3 フォルダを走査して:
- 16kHz mono wav へ変換
- WebRTC VAD で発話区間を検出・分割
- クリップ wav を保存
- メタデータ CSV (segments.csv) を出力

追加機能:
- 各ファイルの冒頭 N 秒を無視 (skip_head_sec)
- 各ファイルの末尾 N 秒を無視 (skip_tail_sec)
- 1ファイルあたりの最大セグメント数 (max_segments_per_file)
- クリップの音量正規化 (normalize)

使い方（CLI例）:
    python dataset_audio_prep.py input_mp3_dir output_dataset_dir
"""

import os
import sys
import csv
import uuid
import wave
import contextlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

import webrtcvad
from pydub import AudioSegment


@dataclass
class VadConfig:
    aggressiveness: int = 2      # 0〜3 (数値大きいほど厳しめ)
    frame_ms: int = 30           # 10, 20, 30 のいずれか
    padding_ms: int = 300        # 前後に付ける余白
    min_segment_ms: int = 500    # 最小長さ (ms)
    max_segment_ms: int = 10000  # 最大長さ (ms)

    # 便利機能:
    skip_head_sec: float = 0.0           # 各ファイルの冒頭をスキップ (秒)
    skip_tail_sec: float = 0.0           # 各ファイルの末尾をスキップ (秒)
    max_segments_per_file: int = 0       # 0 の場合は制限なし
    normalize: bool = False              # クリップを正規化するかどうか


def mp3_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    """
    mp3 を 16kHz, mono, 16-bit PCM の wav に変換する。
    """
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    audio.export(dst_path, format="wav")


def read_wave(path: str) -> Tuple[bytes, int]:
    """
    VAD 用に 16kHz mono 16-bit PCM の wav を読み込む。
    戻り値: (PCMバイト列, サンプリングレート)
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "VAD 用 wav は mono である必要があります"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "VAD 用 wav は 16-bit PCM である必要があります"
        sample_rate = wf.getframerate()
        assert sample_rate == 16000, "VAD 用 wav は 16kHz である必要があります"

        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """
    VAD 用のフレームを作るジェネレータ。
    """
    n_bytes_per_sample = 2
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * n_bytes_per_sample)
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0

    while offset + frame_size <= len(audio):
        yield audio[offset:offset + frame_size], timestamp, duration
        timestamp += duration
        offset += frame_size


def vad_split_segments(
    wav_path: str,
    cfg: VadConfig,
) -> List[Tuple[float, float]]:
    """
    WebRTC VAD を使って発話区間を検出し、
    (start_sec, end_sec) のリストを返す。

    - 冒頭 skip_head_sec 秒以前のセグメントは切り捨て
    - 末尾 skip_tail_sec 秒以降のセグメントは切り捨て
    - min/max_segment_ms で長さフィルタ
    """
    audio, sample_rate = read_wave(wav_path)

    # 全体の長さ（秒）
    total_dur_sec = len(audio) / (2 * sample_rate)

    vad = webrtcvad.Vad(cfg.aggressiveness)
    frames = list(frame_generator(cfg.frame_ms, audio, sample_rate))
    voiced_flags = [vad.is_speech(f[0], sample_rate) for f in frames]

    frame_duration = cfg.frame_ms / 1000.0

    segments_frame = []
    start_idx: Optional[int] = None

    for i, is_voiced in enumerate(voiced_flags):
        if is_voiced and start_idx is None:
            start_idx = i
        elif not is_voiced and start_idx is not None:
            end_idx = i  # this frame は無音
            segments_frame.append((start_idx, end_idx))
            start_idx = None

    if start_idx is not None:
        segments_frame.append((start_idx, len(frames)))

    padded_segments_sec: List[Tuple[float, float]] = []
    pad_frames = int(cfg.padding_ms / cfg.frame_ms)

    for s, e in segments_frame:
        s_padded = max(0, s - pad_frames)
        e_padded = min(len(frames), e + pad_frames)

        start_time = s_padded * frame_duration
        end_time = e_padded * frame_duration

        # ==== 冒頭スキップ ====
        if end_time <= cfg.skip_head_sec:
            # このセグメントは完全にスキップ対象より前
            continue
        if start_time < cfg.skip_head_sec:
            # セグメントの頭だけスキップし、その分切り詰める
            start_time = cfg.skip_head_sec

        # ==== 末尾スキップ ====
        if cfg.skip_tail_sec > 0.0:
            tail_limit = max(0.0, total_dur_sec - cfg.skip_tail_sec)
            if start_time >= tail_limit:
                # 全体の残り tail_sec 内なので丸ごと捨てる
                continue
            if end_time > tail_limit:
                # セグメントの末尾だけ tail_limit に合わせて切る
                end_time = tail_limit

        seg_ms = (end_time - start_time) * 1000.0
        if seg_ms < cfg.min_segment_ms:
            continue
        if seg_ms > cfg.max_segment_ms:
            # 必要ならここで更に細かく分割してもよいが、とりあえずスキップ
            continue

        padded_segments_sec.append((start_time, end_time))

    return padded_segments_sec


def cut_and_save_wav(
    src_wav_path: str,
    segments_sec: List[Tuple[float, float]],
    out_dir: str,
    base_id_prefix: str,
    max_segments_per_file: int = 0,
    normalize: bool = False,
) -> List[dict]:
    """
    segments_sec (start_sec, end_sec) ごとに wav を切り出して保存。
    - max_segments_per_file: >0 なら、1ファイルあたりの最大セグメント数
    - normalize: True の場合、各クリップを peak normalize する

    メタ情報(辞書)のリストを返す。
    """
    audio = AudioSegment.from_wav(src_wav_path)
    results = []

    for idx, (start_sec, end_sec) in enumerate(segments_sec):
        if max_segments_per_file > 0 and idx >= max_segments_per_file:
            break

        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        clip = audio[start_ms:end_ms]

        # 音量正規化（クリップ音防止のため -1 dBFS くらいに）
        if normalize:
            if clip.max_dBFS != float("-inf"):
                gain = -1.0 - clip.max_dBFS  # 最大ピークを -1 dBFS に
                clip = clip.apply_gain(gain)

        utt_id = f"{base_id_prefix}_{idx:04d}"
        rel_path = os.path.join("wavs", f"{utt_id}.wav")
        out_path = os.path.join(out_dir, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        clip.export(out_path, format="wav")

        results.append(
            {
                "utt_id": utt_id,
                "relpath": rel_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "src_wav": os.path.relpath(src_wav_path, out_dir),
            }
        )

    return results


def process_mp3_folder(
    input_dir: str,
    output_dir: str,
    cfg: VadConfig,
    progress_callback=None,
):
    """
    input_dir 以下の mp3 をすべて処理。
    - まず 16kHz mono wav に変換
    - VAD で分割
    - clip wav + segments.csv 生成
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "segments.csv")

    fieldnames = ["utt_id", "relpath", "start_sec", "end_sec", "src_wav"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for root, _, files in os.walk(input_dir):
            for fname in files:
                if not fname.lower().endswith(".mp3"):
                    continue

                src_mp3 = os.path.join(root, fname)

                if progress_callback:
                    progress_callback(f"Converting to wav: {src_mp3}")

                wav_id = uuid.uuid4().hex[:8]
                base_wav_name = f"{os.path.splitext(fname)[0]}_{wav_id}.wav"
                dst_wav_path = os.path.join(output_dir, "converted_wav", base_wav_name)
                os.makedirs(os.path.dirname(dst_wav_path), exist_ok=True)

                mp3_to_wav_16k_mono(src_mp3, dst_wav_path)

                # VAD 分割
                if progress_callback:
                    progress_callback(f"Running VAD: {dst_wav_path}")

                segments = vad_split_segments(dst_wav_path, cfg)
                base_prefix = os.path.splitext(base_wav_name)[0]

                # クリップ保存（max_segments_per_file / normalize 対応）
                clip_meta = cut_and_save_wav(
                    dst_wav_path,
                    segments,
                    output_dir,
                    base_prefix,
                    max_segments_per_file=cfg.max_segments_per_file,
                    normalize=cfg.normalize,
                )

                for row in clip_meta:
                    writer.writerow(row)

                if progress_callback:
                    progress_callback(
                        f"Done: {fname} ({len(clip_meta)} segments)\n"
                    )


# CLI 実行用
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dataset_audio_prep.py <input_mp3_dir> <output_dataset_dir>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    cfg = VadConfig(
        aggressiveness=2,
        frame_ms=30,
        padding_ms=300,
        min_segment_ms=500,
        max_segment_ms=10000,
        skip_head_sec=0.0,         # CLI からはとりあえずデフォルト値
        skip_tail_sec=0.0,
        max_segments_per_file=0,
        normalize=False,
    )

    def simple_logger(msg: str):
        print(msg)

    process_mp3_folder(in_dir, out_dir, cfg, progress_callback=simple_logger)
    print("All done.")
