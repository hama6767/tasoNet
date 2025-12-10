#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コンテナ内で GPT-SoVITS の学習パイプラインを実行するスクリプト。

ステージ:
  1a: 1-get-text.py
  1b: 2-get-hubert-wav32k.py
  1c: 3-get-semantic.py
  S2: s2_train_v3.py
  S1: s1_train.py

使い方（コンテナ内から）例:

  python /workspace/gpt_sovits_train_inner.py \
    --list /host_ws/lists/gpt_sovits_train.list \
    --exp-name streamer_ja \
    --output-root /host_ws/features/streamer_ja \
    --s2-config /host_ws/configs/s2_streamer_ft.json \
    --gpus 0 \
    --stages 1a,1b,1c,s2
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


def run_cmd(cmd, env: Dict[str, str], cwd: Path):
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (code={proc.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True,
                    help="train 用 .list (wav|spk|lang|text)")
    ap.add_argument("--exp-name", type=str, required=True,
                    help="実験名 (1-get-text 等で使われる exp_name)")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="特徴量出力ディレクトリ (opt_dir に渡す)")

    ap.add_argument("--gpus", type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES / _CUDA_VISIBLE_DEVICES に渡す GPU ID 列 (例: 0,1)")
    ap.add_argument("--half", action="store_true", default=False,
                    help="1a/1b/1c を half 精度で実行するか")

    ap.add_argument("--bert-dir", type=Path, default=None,
                    help="BERT モデルディレクトリ (1-get-text 用、未指定ならデフォルト)")
    ap.add_argument("--cnhubert-dir", type=Path, default=None,
                    help="CNHubert ベースディレクトリ (2-get-hubert 用、未指定ならデフォルト)")
    ap.add_argument("--semantic-s2-ckpt", type=Path, default=None,
                    help="3-get-semantic 用 SoVITS ckpt (pretrained_s2G として渡す)")
    ap.add_argument("--semantic-s2-config", type=Path, default=None,
                    help="3-get-semantic 用 SoVITS config (s2config_path として渡す)")

    ap.add_argument("--s2-config", type=Path, default=None,
                    help="SoVITS 学習用 config JSON (s2_train_v3.py の -c)")
    ap.add_argument("--s1-config", type=Path, default=None,
                    help="GPT(Stage1) 学習用 config YAML (s1_train.py の -c)")

    ap.add_argument("--stages", type=str, default="1a,1b,1c,s2",
                    help="実行するステージをカンマ区切りで指定 (例: 1a,1b,1c,s2,s1)")

    args = ap.parse_args()

    repo_dir = Path("/workspace/GPT-SoVITS").resolve()
    prep_dir = repo_dir / "GPT_SoVITS" / "prepare_datasets"
    gpt_root = repo_dir / "GPT_SoVITS"

    if not repo_dir.is_dir():
        raise RuntimeError(f"GPT-SoVITS repo が見つかりません: {repo_dir}")

    list_path = args.list.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stages = set(s.strip() for s in args.stages.split(",") if s.strip())

    env = os.environ.copy()

    # PYTHONPATH: repo 直下 + GPT_SoVITS
    prev_pp = env.get("PYTHONPATH", "")
    extra_paths = [str(repo_dir), str(gpt_root)]
    if prev_pp:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + [prev_pp])
    else:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    env["inp_text"] = str(list_path)
    env["exp_name"] = args.exp_name
    env["opt_dir"] = str(output_root)
    env["i_part"] = "0"
    env["all_parts"] = "1"
    env["_CUDA_VISIBLE_DEVICES"] = args.gpus
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["is_half"] = "True" if args.half else "False"

    # ★ ここがポイント：BERT のデフォルトパス
    default_bert_dir = repo_dir / "GPT-SoVITS" / "pretrained_models" / "bert-base-multilingual-cased"
    if args.bert_dir is not None:
        env["bert_pretrained_dir"] = str(args.bert_dir.resolve())
    else:
        env["bert_pretrained_dir"] = str(default_bert_dir)

    # CN-Hubert は 2-get-hubert 用。未指定なら空文字で渡す（あれば後で同様にデフォルトを生やしてもOK）
    if args.cnhubert_dir is not None:
        env["cnhubert_base_dir"] = str(args.cnhubert_dir.resolve())
    else:
        env["cnhubert_base_dir"] = ""

    python_bin = sys.executable

    # 1a: 1-get-text.py
    if "1a" in stages:
        script = prep_dir / "1-get-text.py"
        if not script.is_file():
            raise FileNotFoundError(f"1-get-text.py が見つかりません: {script}")
        print("[STEP] 1a: 1-get-text.py", flush=True)
        run_cmd([python_bin, str(script)], env=env, cwd=repo_dir)

    # 1b: 2-get-hubert-wav32k.py
    if "1b" in stages:
        script = prep_dir / "2-get-hubert-wav32k.py"
        if not script.is_file():
            raise FileNotFoundError(f"2-get-hubert-wav32k.py が見つかりません: {script}")
        print("[STEP] 1b: 2-get-hubert-wav32k.py", flush=True)
        run_cmd([python_bin, str(script)], env=env, cwd=repo_dir)

    # 1c: 3-get-semantic.py
    if "1c" in stages:
        script = prep_dir / "3-get-semantic.py"
        if not script.is_file():
            raise FileNotFoundError(f"3-get-semantic.py が見つかりません: {script}")

        # script-specific env
        env_sem = env.copy()
        if args.semantic_s2_ckpt is not None:
            env_sem["pretrained_s2G"] = str(args.semantic_s2_ckpt.resolve())
        if args.semantic_s2_config is not None:
            env_sem["s2config_path"] = str(args.semantic_s2_config.resolve())

        print("[STEP] 1c: 3-get-semantic.py", flush=True)
        run_cmd([python_bin, str(script)], env=env_sem, cwd=repo_dir)

    # S2: SoVITS 学習
    if "s2" in stages:
        if args.s2_config is None:
            raise RuntimeError("S2 を実行するには --s2-config を指定してください。")
        script = repo_dir / "GPT_SoVITS" / "s2_train.py"
        if not script.is_file():
            raise FileNotFoundError(f"s2_train.py が見つかりません: {script}")
        print("[STEP] S2: s2_train.py", flush=True)
        run_cmd(
            [python_bin, str(script), "-c", str(args.s2_config.resolve())],
            env=env,
            cwd=repo_dir,
        )

    # S1: GPT 学習
    if "s1" in stages:
        if args.s1_config is None:
            raise RuntimeError("S1 を実行するには --s1-config を指定してください。")
        script = repo_dir / "GPT_SoVITS" / "s1_train.py"
        if not script.is_file():
            raise FileNotFoundError(f"s1_train.py が見つかりません: {script}")
        print("[STEP] S1: s1_train.py", flush=True)
        run_cmd(
            [python_bin, str(script), "-c", str(args.s1_config.resolve())],
            env=env,
            cwd=repo_dir,
        )

    print("[INFO] すべてのステージが完了しました。", flush=True)


if __name__ == "__main__":
    main()
