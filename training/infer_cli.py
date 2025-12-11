#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import types

import numpy as np
import soundfile as sf
import torch


# --- パス調整 --------------------------------------------------------------

def _get_repo_root() -> Path:
    """
    GPT-SoVITS リポジトリのルートを返す。
    - PyInstaller onefile の場合: sys._MEIPASS/GPT-SoVITS
    - 通常の Python 実行: このファイルの親ディレクトリ or フォールバック絶対パス
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

    # PyInstaller で --add-data "training/GPT-SoVITS:GPT-SoVITS" 前提
    candidate = base / "GPT-SoVITS"
    if candidate.exists():
        return candidate.resolve()

    # 通常の clone であれば、このファイルの親ディレクトリがリポジトリルートのはず
    if (base / "GPT_SoVITS").exists():
        return base.resolve()

    # フォールバック: 元の絶対パス（環境依存）
    fallback = Path("/home/hama6767/DA/tasoNet/training/GPT-SoVITS").resolve()
    return fallback


REPO_ROOT = _get_repo_root()
GPT_ROOT = REPO_ROOT / "GPT_SoVITS"

for p in (REPO_ROOT, GPT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


# --- グローバル: inference_webui モジュール / i18n -----------------------

TTS: Any = None
I18N: Any = None


def disable_f5_tts():
    """
    f5_tts / f5_tts.model をダミーモジュールとして sys.modules に登録する。
    """
    dummy_pkg = types.ModuleType("f5_tts")
    dummy_model = types.ModuleType("f5_tts.model")

    class DummyDiT:
        pass

    dummy_model.DiT = DummyDiT
    dummy_pkg.model = dummy_model

    sys.modules["f5_tts"] = dummy_pkg
    sys.modules["f5_tts.model"] = dummy_model

def disable_sv():
    """
    sv モジュール（話者認識）をダミーに差し替える。

    GPT_SoVITS/inference_webui.py は:
        from sv import SV
    を実行するが、今回の CLI / GUI では SV 機能を使わないので、
    本物の sv.py を import せず、ここで用意したダミーの SV クラスを渡す。
    """

    dummy_mod = types.ModuleType("sv")

    class DummySV(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # 本当は話者埋め込みを扱うが、今回は使わない

        def set_spkemb(self, *args, **kwargs):
            # 何もしないダミー
            pass

        def get_spkemb(self, *args, **kwargs):
            # 適当なダミー埋め込み（192次元ゼロベクトルなど）を返しておく
            return torch.zeros(192)

    dummy_mod.SV = DummySV
    sys.modules["sv"] = dummy_mod

def disable_torchcodec_for_torchaudio():
    """
    torchaudio.load が torchcodec の AudioDecoder を使うのをやめさせて、
    soundfile ベースのシンプルな loader に差し替える。

    これにより、
      RuntimeError: Failed to create AudioDecoder for <wav>: <JSONエラー>
    を回避する。
    """
    try:
        import torchaudio  # すでにインストールされている前提
    except Exception as e:
        print(f"[warn] torchaudio import failed, skip patch: {e}")
        return

    import soundfile as sf
    import numpy as _np
    import torch as _torch

    def _load_with_soundfile(
        uri,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format=None,
        buffer_size: int = 0,
        backend=None,
    ):
        # soundfile で WAV を読む（常に 2D [time, ch]）
        data, sr = sf.read(uri, always_2d=True)

        # frame_offset / num_frames を反映
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]

        # torchaudio.load と同じく float32 に正規化しておく
        if data.dtype.kind in ("i", "u"):
            max_val = _np.iinfo(data.dtype).max
            data = data.astype("float32") / float(max_val)
        else:
            data = data.astype("float32")

        if channels_first:
            tensor = _torch.from_numpy(data.T)  # [ch, time]
        else:
            tensor = _torch.from_numpy(data)    # [time, ch]

        return tensor, sr

    torchaudio.load = _load_with_soundfile
    print("[info] patched torchaudio.load to use soundfile (torchcodec AudioDecoder 無効化)")


# --- ユーティリティ --------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GPT-SoVITS CLI: text + ref wav -> wav (s1v3 + s2G488k 前提)"
    )

    ap.add_argument("--config", type=str, required=True,
                    help="SoVITS 用 config (s2.json など)")
    ap.add_argument("--s2-ckpt", type=str, required=True,
                    help="SoVITS G の checkpoint (s2Gxxx.pth)")
    ap.add_argument("--gpt-ckpt", type=str, required=True,
                    help="GPT(Stage1) の checkpoint (s1v3.ckpt など)")

    ap.add_argument("--ref-wav", type=str, required=True,
                    help="リファレンス話者の wav (32kHz mono 推奨)")
    ap.add_argument("--text", type=str, required=True,
                    help="合成したいテキスト")
    ap.add_argument("--prompt-text", type=str, default=None,
                    help="ref-wav に対応するテキスト。未指定なら --text を流用")

    ap.add_argument("--out-wav", type=str, required=True,
                    help="出力 wav パス (ディレクトリは事前に作っておく)")

    ap.add_argument("--bert-path", type=str, default=None,
                    help="BERT モデルパス or モデルID (未指定なら bert-base-multilingual-cased)")
    ap.add_argument("--cnhubert-path", type=str, default=None,
                    help="CN-HuBERT のベースディレクトリ")

    ap.add_argument("--no-half", action="store_true",
                    help="True: fp32 で推論 (BigVGAN との整合のため推奨)")

    return ap.parse_args()


def _setup_env(
    bert_path: Optional[str],
    cnhubert_path: Optional[str],
    no_half: bool,
    gpt_ckpt: Optional[str],
) -> None:
    """
    環境変数を inference_webui と揃えておく。

    ※ 重要:
      - inference_webui は BERT を HuggingFace の「モデルID」として扱う前提で書かれている。
      - ローカルディレクトリパスを渡すと今回のような HFValidationError を起こしやすい。
      - そのため、パスっぽい値は無視して 'bert-base-multilingual-cased' を使う。
    """
    from pathlib import Path

    # fast-langdetect 用キャッシュディレクトリ
    fast_cache = GPT_ROOT / "pretrained_models" / "fast_langdetect"
    fast_cache.mkdir(parents=True, exist_ok=True)
    os.environ["FAST_LANGDETECT_CACHE"] = str(fast_cache)

    # ---------- BERT ----------
    use_id: str

    if bert_path:
        # 「/」や「\」を含む＝ファイルパスっぽいと判断
        if ("/" in bert_path) or ("\\" in bert_path) or Path(bert_path).exists():
            # ローカルディレクトリは inference_webui 側と相性が悪いのでやめる
            use_id = "bert-base-multilingual-cased"
            print(
                "[warn] BERT パスがローカルパスに見えたため、"
                "HuggingFace モデルID 'bert-base-multilingual-cased' にフォールバックします。"
            )
        else:
            # スラッシュを含まない文字列は「モデルID」とみなす
            use_id = bert_path
            print(f"[info] using BERT model id: {use_id}")
    else:
        use_id = "bert-base-multilingual-cased"
        print("[info] using BERT model id: bert-base-multilingual-cased")

    os.environ["bert_path"] = use_id

    # ---------- CN-HuBERT ----------
    if cnhubert_path:
        os.environ["cnhubert_base_path"] = cnhubert_path
        print(f"[info] cnhubert_base_path = {cnhubert_path}")
    else:
        default_cnhubert = GPT_ROOT / "pretrained_models" / "chinese-hubert-base"
        os.environ["cnhubert_base_path"] = str(default_cnhubert)
        print(f"[info] cnhubert_base_path = {default_cnhubert}")

    # half / full
    os.environ["is_half"] = "False" if no_half else "True"

    # GPT ckpt（inference_webui が参照する）
    if gpt_ckpt:
        os.environ["gpt_path"] = gpt_ckpt
        print(f"[info] gpt_path = {gpt_ckpt}")



def import_tts_modules() -> None:
    """
    GPT_SoVITS/inference_webui.py を import して TTS / I18N を初期化
    """
    global TTS, I18N
    if TTS is not None:
        return

    from GPT_SoVITS import inference_webui as tts_mod

    TTS = tts_mod
    I18N = tts_mod.i18n
    print("[info] TTS modules imported.")


def load_s2_model(config_path: str, ckpt_path: str, device: torch.device) -> Tuple[Any, Any]:
    """
    SoVITS(S2) モデルを inference_webui の change_sovits_weights 経由でロード。
    """
    assert TTS is not None

    from GPT_SoVITS import utils

    hps_from_file = utils.get_hparams_from_file(config_path)
    print(f"[info] config sr = {hps_from_file.data.sampling_rate}")

    try:
        rv = TTS.change_sovits_weights(ckpt_path)
        # generator の場合だけ 1 ステップ回す
        if rv is not None and hasattr(rv, "__iter__") and not isinstance(rv, (tuple, list, dict)):
            try:
                next(rv)
            except StopIteration:
                pass
    except UnboundLocalError as e:
        print(f"[warn] change_sovits_weights raised {e!r}, ignoring (assuming SoVITS is loaded).",
              file=sys.stderr)
    except Exception as e:
        raise RuntimeError(f"failed to load SoVITS weights from {ckpt_path}: {e}") from e

    model = getattr(TTS, "vq_model", None)
    hps = getattr(TTS, "hps", None)

    if model is None or hps is None:
        raise RuntimeError("SoVITS model or hps is not set after change_sovits_weights().")

    print("[info] SoVITS loaded.")
    return model, hps


def pick_lang_key_for_code(target_code: str) -> str:
    """
    inference_webui.dict_language の中から、value が target_code (例: 'all_ja') に
    対応する key を返す。見つからなければ最初の key。
    """
    assert TTS is not None
    dict_language = TTS.dict_language

    for k, v in dict_language.items():
        if v == target_code:
            return k
    return next(iter(dict_language.keys()))


def extract_features(
    ref_wav_path: str,
    text: str,
    prompt_text: Optional[str] = None,
    *,
    language_code: str = "all_ja",
) -> Dict[str, Any]:
    """
    get_tts_wav に渡すパラメータ一式を dict にまとめる。
    言語は「全部日本語 (all_ja)」を想定。
    """
    assert I18N is not None
    if not text.strip():
        raise ValueError("text が空です。何かしゃべらせたいテキストを指定してください。")

    # ref テキスト（未指定なら text を流用）
    if prompt_text is None or not prompt_text.strip():
        prompt_text = text

    ja_lang_key = pick_lang_key_for_code(language_code)

    features: Dict[str, Any] = {
        "ref_wav_path": ref_wav_path,
        "prompt_text": prompt_text,
        "prompt_language": ja_lang_key,
        "text": text,
        "text_language": ja_lang_key,
        "how_to_cut": I18N("不切"),
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "ref_free": False,
        "speed": 1.0,
        "if_freeze": False,
    }
    return features


def synthesize(
    model: Any,
    hps: Any,
    features: Dict[str, Any],
) -> Tuple[int, np.ndarray]:
    """
    実際の TTS 実行。最後のチャンクを最終出力として返す。
    """
    assert TTS is not None

    gen = TTS.get_tts_wav(
        ref_wav_path=features["ref_wav_path"],
        prompt_text=features["prompt_text"],
        prompt_language=features["prompt_language"],
        text=features["text"],
        text_language=features["text_language"],
        how_to_cut=features["how_to_cut"],
        top_k=features["top_k"],
        top_p=features["top_p"],
        temperature=features["temperature"],
        ref_free=features["ref_free"],
        speed=features["speed"],
        if_freeze=features["if_freeze"],
    )

    last_sr: Optional[int] = None
    last_audio: Optional[np.ndarray] = None

    for sr, audio in gen:
        last_sr = sr
        last_audio = audio

    if last_sr is None or last_audio is None:
        raise RuntimeError("get_tts_wav did not yield any audio")

    if not isinstance(last_audio, np.ndarray):
        last_audio = np.asarray(last_audio)

    if last_audio.dtype != np.int16:
        last_audio = last_audio.astype(np.int16)

    return int(last_sr), last_audio


def save_wav(sr: int, audio: np.ndarray, out_path: str) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if audio.dtype == np.int16:
        wav = audio.astype(np.float32) / 32768.0
    else:
        wav = audio.astype(np.float32)

    sf.write(str(out_p), wav, sr)
    print(f"[info] saved: {out_p} (sr={sr}, dur={len(wav) / sr:.2f}s)")


# --- InferenceSession ------------------------------------------------------


class InferenceSession:
    """
    GPT-SoVITS の推論セッション。
    - inference_webui / SoVITS / GPT の読み込みを 1 回だけ行い、
      同一プロセス内で複数回の推論を回せるようにする。
    """

    def __init__(
        self,
        config_path: str,
        s2_ckpt_path: str,
        gpt_ckpt_path: Optional[str],
        *,
        bert_path: Optional[str] = None,
        cnhubert_path: Optional[str] = None,
        no_half: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        disable_f5_tts()
        disable_sv()
        disable_torchcodec_for_torchaudio() 

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.config_path = str(config_path)
        self.bert_path = bert_path
        self.cnhubert_path = cnhubert_path
        self.no_half = no_half
        self.gpt_ckpt_path = str(gpt_ckpt_path) if gpt_ckpt_path else None

        print(f"[info] device = {self.device.type}")

        _setup_env(self.bert_path, self.cnhubert_path, self.no_half, self.gpt_ckpt_path)
        import_tts_modules()

        # SoVITS ロード
        self.s2_model: Any
        self.hps: Any
        self.current_s2_ckpt: Optional[str] = None
        self.change_s2_ckpt(str(s2_ckpt_path), config_path=self.config_path)

        # GPT ロード
        if self.gpt_ckpt_path:
            self.change_gpt_ckpt(self.gpt_ckpt_path)
        else:
            print("[info] gpt_ckpt_path が指定されていないため、inference_webui のデフォルト設定に任せます。")

    # --- モデル差し替え系 ---

    def change_s2_ckpt(self, s2_ckpt_path: str, *, config_path: Optional[str] = None) -> None:
        """
        SoVITS の ckpt を差し替える。
        config_path が指定されていればそれも更新する。
        """
        s2_ckpt_path = str(Path(s2_ckpt_path).resolve())
        if self.current_s2_ckpt == s2_ckpt_path and (config_path is None or config_path == self.config_path):
            return

        if config_path is not None:
            self.config_path = str(config_path)

        print(f"[info] loading SoVITS from {s2_ckpt_path}")
        self.s2_model, self.hps = load_s2_model(self.config_path, s2_ckpt_path, self.device)
        self.current_s2_ckpt = s2_ckpt_path

    def change_gpt_ckpt(self, gpt_ckpt_path: str) -> None:
        """
        GPT の ckpt を差し替える。
        """
        gpt_ckpt_path = str(Path(gpt_ckpt_path).resolve())
        self.gpt_ckpt_path = gpt_ckpt_path
        os.environ["gpt_path"] = gpt_ckpt_path
        print(f"[info] loading GPT weights from {gpt_ckpt_path}")
        assert TTS is not None
        TTS.change_gpt_weights(gpt_ckpt_path)

    # --- 推論 ---

    def infer(
        self,
        ref_wav_path: str,
        text: str,
        prompt_text: Optional[str] = None,
        *,
        language_code: str = "all_ja",
    ) -> Tuple[int, np.ndarray]:
        """
        1 回分の音声合成を実行し、(sr, audio) を返す。
        """
        feat = extract_features(
            ref_wav_path=ref_wav_path,
            text=text,
            prompt_text=prompt_text,
            language_code=language_code,
        )
        sr, audio = synthesize(self.s2_model, self.hps, feat)
        return sr, audio


# --- CLI エントリポイント -------------------------------------------------


def main() -> None:
    args = parse_args()

    session = InferenceSession(
        config_path=args.config,
        s2_ckpt_path=args.s2_ckpt,
        gpt_ckpt_path=args.gpt_ckpt,
        bert_path=args.bert_path,
        cnhubert_path=args.cnhubert_path,
        no_half=args.no_half,
    )

    prompt_text = args.prompt_text if args.prompt_text is not None else args.text

    sr, audio = session.infer(
        ref_wav_path=args.ref_wav,
        text=args.text,
        prompt_text=prompt_text,
    )

    save_wav(sr, audio, args.out_wav)


if __name__ == "__main__":
    main()
