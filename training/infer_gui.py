#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-SoVITS 推論専用 GUI（Docker / 外部 Python 不要版）

- 同梱された GPT-SoVITS ディレクトリと BERT を使って infer_cli.py を直接叩く。
- PyInstaller でビルドしたときは、_MEIPASS 配下の GPT-SoVITS を参照する。
"""


import os
import sys
import traceback
import io
import contextlib
import subprocess

import site
from pathlib import Path
from typing import Optional

# ★ ここを追加：ユーザー site-packages を先頭に追加する
user_site = os.path.expanduser(
    f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
if os.path.isdir(user_site):
    site.addsitedir(user_site)

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QMessageBox,
    QCheckBox,
)

REPO_DIR = Path("/home/hama6767/DA/tasoNet/training/GPT-SoVITS").resolve()
os.chdir(REPO_DIR)
print(f"[GUI] chdir to {REPO_DIR}")

# ===== パス解決ユーティリティ =====

def get_base_dir() -> Path:
    """
    実行ファイルのベースディレクトリを返す。
    - 生Python実行時: この .py のあるフォルダ
    - PyInstaller exe: _MEIPASS (展開先)
    """
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def get_repo_dir() -> Path:
    """
    GPT-SoVITS のルートディレクトリ。

    PyInstaller では --add-data "training/GPT-SoVITS:GPT-SoVITS"
    でバンドルする前提。
    """
    base = get_base_dir()
    # exe の中では base/GPT-SoVITS に展開される
    repo = base / "GPT-SoVITS"

    # 生Pythonで手元から動かしたい場合のフォールバック
    if not repo.exists():
        repo = Path("/home/hama6767/DA/tasoNet/training/GPT-SoVITS").resolve()

    return repo


def get_infer_cli_path() -> Path:
    return get_repo_dir() / "infer_cli.py"


def get_default_bert_dir() -> Path:
    # GPT-SoVITS/GPT_SoVITS/pretrained_models/bert-base-multilingual-cased
    return get_repo_dir() / "GPT_SoVITS" / "pretrained_models" / "bert-base-multilingual-cased"


def get_default_s2_config() -> Path:
    # GPT-SoVITS/GPT_SoVITS/configs/s2.json
    return get_repo_dir() / "GPT_SoVITS" / "configs" / "s2.json"


# ===== infer_cli.py の動的 import ラッパ =====


def run_infer_cli(args_list):
    """
    infer_cli.py を別プロセスとして実行する。
    - cmd: [sys.executable, infer_cli.py, <args...>]
    - stdout / stderr をひとまとめにして返す
    戻り値: (return_code, output_str)
    """
    infer_cli_path = get_infer_cli_path()

    cmd = [sys.executable, str(infer_cli_path)] + args_list

    # REPO_DIR を cwd にすることで、bash からの実行と同じ条件に合わせる
    cwd = str(get_repo_dir())

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
    )

    output_lines = []

    if proc.stdout is not None:
        for line in proc.stdout:
            output_lines.append(line)

    rc = proc.wait()
    return rc, "".join(output_lines)




# ===== 推論スレッド =====

class InferWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal(int)

    def __init__(
        self,
        text: str,
        prompt_text: str,
        ref_wav: Path,
        out_wav: Path,
        s2_ckpt: Path,
        gpt_ckpt: Optional[Path],
        config_path: Optional[Path],
        bert_path: Optional[Path],
        no_half: bool,
    ):
        super().__init__()
        self.text = text
        self.prompt_text = prompt_text
        self.ref_wav = ref_wav
        self.out_wav = out_wav
        self.s2_ckpt = s2_ckpt
        self.gpt_ckpt = gpt_ckpt
        self.config_path = config_path
        self.bert_path = bert_path
        self.no_half = no_half

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def log_env_debug(self):
        """デバッグ用に、実行環境の情報をログに吐く"""
        try:
            self.log("[DEBUG] ----- ENV DUMP START -----")
            self.log(f"[DEBUG] cwd          : {os.getcwd()}")
            self.log(f"[DEBUG] sys.executable: {sys.executable}")

            self.log("[DEBUG] sys.path (first 10):")
            for p in sys.path[:10]:
                self.log("  - " + p)

            for key in ("gpt_path", "cnhubert_base_path", "bert_path", "is_half", "version"):
                if key in os.environ:
                    self.log(f"[DEBUG] env {key} = {os.environ[key]}")

            self.log("[DEBUG] ----- ENV DUMP END -----")
        except Exception as e:
            # ここでコケても本体推論には影響させない
            self.log(f"[DEBUG] log_env_debug でエラー: {e}")

    def run(self):
        args = []

        # config / BERT / GPT は、指定があれば infer_cli に渡す
        if self.config_path is not None:
            args += ["--config", str(self.config_path.resolve())]
        if self.bert_path is not None:
            args += ["--bert-path", str(self.bert_path.resolve())]
        if self.gpt_ckpt is not None:
            args += ["--gpt-ckpt", str(self.gpt_ckpt.resolve())]

        args += [
            "--s2-ckpt",
            str(self.s2_ckpt.resolve()),
            "--ref-wav",
            str(self.ref_wav.resolve()),
            "--text",
            self.text,
            "--out-wav",
            str(self.out_wav.resolve()),
        ]

        if self.prompt_text.strip():
            args += ["--prompt-text", self.prompt_text]

        if self.no_half:
            args.append("--no-half")

        self.log("[INFO] infer_cli.py を実行します")
        self.log("[INFO] args = " + " ".join(args))

        # ★ 実行環境の情報を吐く
        self.log_env_debug()

        rc = 0
        try:
            # ★ サブプロセスで infer_cli.py を実行
            rc, cli_output = run_infer_cli(args)

            # infer_cli の標準出力をログに流す
            if cli_output and cli_output.strip():
                self.log("[INFO] --- infer_cli stdout/stderr ---")
                for line in cli_output.splitlines():
                    self.log("[infer_cli] " + line)
                self.log("[INFO] --- infer_cli end ---")

            if rc != 0:
                self.log(f"[ERROR] infer_cli.py の終了コード: {rc}")

        except Exception:
            tb = traceback.format_exc()
            self.log("[ERROR] 推論中に例外が発生しました (フルスタックトレース):")
            self.log(tb)
            rc = -1

        self.done_signal.emit(rc)



# ===== GUI 本体 =====

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS Inference GUI (Standalone + BERT 内蔵)")
        self.resize(900, 700)

        central = QWidget()
        layout = QVBoxLayout(central)

        # モデルパス設定
        model_layout = QVBoxLayout()

        s2_layout = QHBoxLayout()
        s2_layout.addWidget(QLabel("SoVITS ckpt (.pth):"))
        # ★ ユーザーの FT 重み (外部ファイル)：ここは自分のパスに合わせて変更してOK
        self.s2_edit = QLineEdit(
            "/home/hama6767/DA/tasoNet/SoVITS_weights_v2/exp_streamer_e8_s8288.pth"
        )
        s2_btn = QPushButton("参照...")
        s2_btn.clicked.connect(self.browse_s2_ckpt)
        s2_layout.addWidget(self.s2_edit)
        s2_layout.addWidget(s2_btn)
        model_layout.addLayout(s2_layout)

        gpt_layout = QHBoxLayout()
        gpt_layout.addWidget(QLabel("GPT ckpt (任意 / 空で infer_cli デフォルト):"))
        self.gpt_edit = QLineEdit(
            str(
                get_repo_dir()
                / "GPT_SoVITS"
                / "pretrained_models"
                / "GPT-SoVITS"
                / "gsv-v2final-pretrained"
                / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
            )
        )
        gpt_btn = QPushButton("参照...")
        gpt_btn.clicked.connect(self.browse_gpt_ckpt)
        gpt_layout.addWidget(self.gpt_edit)
        gpt_layout.addWidget(gpt_btn)
        model_layout.addLayout(gpt_layout)

        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("config (任意 / 空で infer_cli デフォルト):"))
        self.cfg_edit = QLineEdit(str(get_default_s2_config()))
        cfg_btn = QPushButton("参照...")
        cfg_btn.clicked.connect(self.browse_config)
        cfg_layout.addWidget(self.cfg_edit)
        cfg_layout.addWidget(cfg_btn)
        model_layout.addLayout(cfg_layout)

        bert_layout = QHBoxLayout()
        bert_layout.addWidget(QLabel("BERT ディレクトリ (同梱されたものを使用):"))
        self.bert_edit = QLineEdit(str(get_default_bert_dir()))
        bert_btn = QPushButton("参照...")
        bert_btn.clicked.connect(self.browse_bert)
        bert_layout.addWidget(self.bert_edit)
        bert_layout.addWidget(bert_btn)
        model_layout.addLayout(bert_layout)

        layout.addLayout(model_layout)

        # no-half
        half_layout = QHBoxLayout()
        self.no_half_checkbox = QCheckBox("--no-half を付ける（FP32 推論）")
        self.no_half_checkbox.setChecked(True)
        half_layout.addWidget(self.no_half_checkbox)
        layout.addLayout(half_layout)

        # 音声 / テキスト
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("参照 wav:"))
        self.ref_edit = QLineEdit("/home/hama6767/DA/tasoNet/tsukuyomi.wav")
        ref_btn = QPushButton("参照...")
        ref_btn.clicked.connect(self.browse_ref_wav)
        ref_layout.addWidget(self.ref_edit)
        ref_layout.addWidget(ref_btn)
        layout.addLayout(ref_layout)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("出力 wav:"))
        self.out_edit = QLineEdit("/home/hama6767/DA/tasoNet/out/test_001.wav")
        out_btn = QPushButton("参照...")
        out_btn.clicked.connect(self.browse_out_wav)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)

        txt_layout = QVBoxLayout()
        txt_layout.addWidget(QLabel("プロンプトテキスト (任意 / 空でもOK):"))
        self.prompt_edit = QTextEdit(
            "また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。"
        )
        txt_layout.addWidget(self.prompt_edit)

        txt_layout.addWidget(QLabel("生成したいテキスト:"))
        self.text_edit = QTextEdit("あまねかなただよ、こんかなた。よろしくね。")
        txt_layout.addWidget(self.text_edit)
        layout.addLayout(txt_layout)

        # 実行ボタン
        self.run_btn = QPushButton("推論実行")
        self.run_btn.clicked.connect(self.start_infer)
        layout.addWidget(self.run_btn)

        # ログ
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setCentralWidget(central)
        self.worker: Optional[InferWorker] = None

    # ---- ブラウズ系 ----
    def browse_s2_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SoVITS ckpt を選択",
            filter="Checkpoint (*.pth *.ckpt);;All files (*)",
        )
        if path:
            self.s2_edit.setText(path)

    def browse_gpt_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "GPT ckpt を選択",
            filter="Checkpoint (*.pth *.ckpt);;All files (*)",
        )
        if path:
            self.gpt_edit.setText(path)

    def browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "config (s2.json 等) を選択",
            filter="JSON files (*.json);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def browse_bert(self):
        path = QFileDialog.getExistingDirectory(self, "BERT ディレクトリを選択")
        if path:
            self.bert_edit.setText(path)

    def browse_ref_wav(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "参照 wav を選択",
            filter="WAV files (*.wav);;All files (*)",
        )
        if path:
            self.ref_edit.setText(path)

    def browse_out_wav(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "出力 wav を保存",
            filter="WAV files (*.wav);;All files (*)",
        )
        if path:
            self.out_edit.setText(path)

    # ---- 実行 ----
    def append_log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def on_done(self, rc: int):
        self.run_btn.setEnabled(True)
        if rc == 0:
            self.append_log("[INFO] 推論が正常に終了しました。")
            QMessageBox.information(self, "完了", "推論が完了しました。")
        else:
            self.append_log(f"[ERROR] 推論が異常終了しました (rc={rc})。")
            QMessageBox.warning(self, "エラー", f"推論がエラー終了しました (rc={rc})。")

    def start_infer(self):
        try:
            cfg = self.build_cfg_from_ui()
        except ValueError as e:
            QMessageBox.warning(self, "入力エラー", str(e))
            return

        self.log_text.clear()
        self.append_log("[INFO] 推論を開始します...")
        self.run_btn.setEnabled(False)

        self.worker = InferWorker(**cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()

    def build_cfg_from_ui(self):
        s2_text = self.s2_edit.text().strip()
        if not s2_text:
            raise ValueError("SoVITS ckpt を指定してください。")
        s2_ckpt = Path(s2_text)
        if not s2_ckpt.is_file():
            raise ValueError("SoVITS ckpt ファイルが存在しません。")

        ref_text = self.ref_edit.text().strip()
        if not ref_text:
            raise ValueError("参照 wav を指定してください。")
        ref_wav = Path(ref_text)
        if not ref_wav.is_file():
            raise ValueError("参照 wav が存在しません。")

        out_text = self.out_edit.text().strip()
        if not out_text:
            raise ValueError("出力 wav のパスを指定してください。")
        out_wav = Path(out_text)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        text = self.text_edit.toPlainText().strip()
        if not text:
            raise ValueError("生成テキストを入力してください。")

        prompt_text = self.prompt_edit.toPlainText()

        gpt_text = self.gpt_edit.text().strip()
        gpt_ckpt = Path(gpt_text) if gpt_text else None

        cfg_text = self.cfg_edit.text().strip()
        config_path = Path(cfg_text) if cfg_text else None

        bert_text = self.bert_edit.text().strip()
        bert_path = Path(bert_text) if bert_text else None

        no_half = self.no_half_checkbox.isChecked()

        return dict(
            text=text,
            prompt_text=prompt_text,
            ref_wav=ref_wav,
            out_wav=out_wav,
            s2_ckpt=s2_ckpt,
            gpt_ckpt=gpt_ckpt,
            config_path=config_path,
            bert_path=bert_path,
            no_half=no_half,
        )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
