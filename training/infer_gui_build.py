#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-SoVITS 推論専用 GUI（Docker / 外部 Python 不要版）

- 同梱された GPT-SoVITS ディレクトリと BERT を使って infer_cli.py を直接叩く。
- PyInstaller でビルドしたときは、_MEIPASS 配下の GPT-SoVITS を参照する。
- インストールディレクトリ直下に models/ref_wav/out を自動作成する。
"""

import os
import sys
import subprocess
import traceback
import site
from pathlib import Path
from typing import Optional
import importlib.util

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

# ===== ユーザー site-packages を追加（ビルド環境に依存したパッケージ用） =====

user_site = os.path.expanduser(
    f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
if os.path.isdir(user_site):
    site.addsitedir(user_site)


# ===== インストールディレクトリ / レイアウト関連 =====

def get_base_dir() -> Path:
    """
    実行ファイルのベースディレクトリを返す。
    - 生Python実行時: この .py のあるフォルダ
    - PyInstaller exe: _MEIPASS (展開先) を優先
    """
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def get_install_root() -> Path:
    """
    「インストールディレクトリ」＝ exe が存在するフォルダ。

    onedir ビルド想定:
      dist/gpt_sovits_gui/
        ├ gpt_sovits_gui (実行ファイル)
        ├ GPT-SoVITS/   (add-data で同梱)
        ├ models/
        ├ ref_wav/
        └ out/
    """
    # _MEIPASS は展開先で、onedir の場合は app フォルダと同じになる想定。
    base = get_base_dir()
    # exe の親ディレクトリ（生Python時は .py の親）
    return Path(sys.argv[0]).resolve().parent if getattr(sys, "frozen", False) else base


def ensure_initial_layout() -> None:
    """
    インストールディレクトリ配下に最低限のフォルダを自動作成。

    - {install_root}/models/sovits    : FT SoVITS ckpt 置き場
    - {install_root}/ref_wav          : 参照音声の置き場
    - {install_root}/out              : 出力 wav の置き場
    """
    root = get_install_root()
    models_dir = root / "models"
    s2_dir = models_dir / "sovits"
    ref_dir = root / "ref_wav"
    out_dir = root / "out"

    for d in (models_dir, s2_dir, ref_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)


def find_default_s2_ckpt() -> Optional[Path]:
    """
    models/sovits 配下の最初の .pth をデフォルト SoVITS ckpt として返す。
    無ければ None。
    """
    s2_dir = get_install_root() / "models" / "sovits"
    if not s2_dir.exists():
        return None
    candidates = sorted(s2_dir.glob("*.pth"))
    return candidates[0] if candidates else None


def find_default_ref_wav() -> Optional[Path]:
    """
    ref_wav 配下の最初の .wav をデフォルト参照 wav として返す。
    無ければ None。
    """
    ref_dir = get_install_root() / "ref_wav"
    if not ref_dir.exists():
        return None
    candidates = sorted(ref_dir.glob("*.wav"))
    return candidates[0] if candidates else None


# 起動時にフォルダを自動作成しておく
ensure_initial_layout()


# ===== GPT-SoVITS リポジトリパス =====

def get_repo_dir() -> Path:
    """
    GPT-SoVITS のルートディレクトリ。

    PyInstaller では --add-data "training/GPT-SoVITS:GPT-SoVITS"
    でバンドルする前提。
    """
    base = get_base_dir()
    repo = base / "GPT-SoVITS"

    # バンドルされた GPT-SoVITS があればそれを使う
    if repo.exists():
        return repo

    # 生Pythonでの開発環境用フォールバック（あなたの元のパス）
    return Path("/home/hama6767/DA/tasoNet/training/GPT-SoVITS").resolve()


def get_infer_cli_path() -> Path:
    return get_repo_dir() / "infer_cli.py"


def get_default_bert_dir() -> Path:
    # GPT-SoVITS/GPT_SoVITS/pretrained_models/bert-base-multilingual-cased
    return get_repo_dir() / "GPT_SoVITS" / "pretrained_models" / "bert-base-multilingual-cased"


def get_default_s2_config() -> Path:
    # GPT-SoVITS/GPT_SoVITS/configs/s2.json
    return get_repo_dir() / "GPT_SoVITS" / "configs" / "s2.json"


# ===== infer_cli.py 実行ラッパ（サブプロセス） =====

def run_infer_cli(args_list):
    """
    infer_cli.py を CLI として実行する。
    - 非 frozen (生 Python) のとき: subprocess で別プロセス起動
    - frozen (PyInstaller exe) のとき: モジュールとして import し main() を直接呼ぶ
    戻り値: (return_code, output_str)
    """
    infer_cli_path = get_infer_cli_path()

    # ---- PyInstaller で凍結された exe の場合 ----
    if getattr(sys, "frozen", False):
        # GPT-SoVITS ルートを cwd に
        repo_dir = str(get_repo_dir())
        old_cwd = os.getcwd()
        os.chdir(repo_dir)

        # infer_cli を import
        spec = importlib.util.spec_from_file_location("gpt_sovits_infer_cli", infer_cli_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        argv_backup = sys.argv[:]

        try:
            # argparse が見る sys.argv を差し替え
            sys.argv = ["infer_cli.py"] + args_list

            # print / エラー出力をキャプチャ
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                module.main()

            out = stdout_buf.getvalue() + stderr_buf.getvalue()
            return 0, out

        except SystemExit as e:
            # argparse が sys.exit() したとき
            code = e.code if isinstance(e.code, int) else 0
            out = stdout_buf.getvalue() + stderr_buf.getvalue()
            return code, out

        finally:
            sys.argv = argv_backup
            os.chdir(old_cwd)

    # ---- 通常の Python 実行時 (開発中など) ----
    cmd = [sys.executable, str(infer_cli_path)] + args_list
    cwd = str(get_repo_dir())  # bash から実行した時と同じ

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
        self.log_env_debug()

        rc = 0
        try:
            rc, cli_output = run_infer_cli(args)

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

        install_root = get_install_root()
        models_s2_dir = install_root / "models" / "sovits"
        ref_dir = install_root / "ref_wav"
        out_dir = install_root / "out"

        # --- モデルパス設定 ---
        model_layout = QVBoxLayout()

        # SoVITS ckpt
        s2_layout = QHBoxLayout()
        s2_layout.addWidget(QLabel("SoVITS ckpt (.pth):"))

        default_s2 = find_default_s2_ckpt()
        self.s2_edit = QLineEdit(str(default_s2) if default_s2 else "")
        if default_s2 is None:
            self.s2_edit.setPlaceholderText(str(models_s2_dir / "your_sovits_model.pth"))

        s2_btn = QPushButton("参照...")
        s2_btn.clicked.connect(self.browse_s2_ckpt)
        s2_layout.addWidget(self.s2_edit)
        s2_layout.addWidget(s2_btn)
        model_layout.addLayout(s2_layout)

        # GPT ckpt
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

        # config
        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("config (任意 / 空で infer_cli デフォルト):"))
        self.cfg_edit = QLineEdit(str(get_default_s2_config()))
        cfg_btn = QPushButton("参照...")
        cfg_btn.clicked.connect(self.browse_config)
        cfg_layout.addWidget(self.cfg_edit)
        cfg_layout.addWidget(cfg_btn)
        model_layout.addLayout(cfg_layout)

        # BERT
        bert_layout = QHBoxLayout()
        bert_layout.addWidget(QLabel("BERT ディレクトリ (同梱されたものを使用):"))
        self.bert_edit = QLineEdit(str(get_default_bert_dir()))
        bert_btn = QPushButton("参照...")
        bert_btn.clicked.connect(self.browse_bert)
        bert_layout.addWidget(self.bert_edit)
        bert_layout.addWidget(bert_btn)
        model_layout.addLayout(bert_layout)

        layout.addLayout(model_layout)

        # --- no-half ---
        half_layout = QHBoxLayout()
        self.no_half_checkbox = QCheckBox("--no-half を付ける（FP32 推論）")
        self.no_half_checkbox.setChecked(True)
        half_layout.addWidget(self.no_half_checkbox)
        layout.addLayout(half_layout)

        # --- 音声 / テキスト ---
        # 参照 wav
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("参照 wav:"))
        default_ref = find_default_ref_wav()
        self.ref_edit = QLineEdit(str(default_ref) if default_ref else "")
        if default_ref is None:
            self.ref_edit.setPlaceholderText(str(ref_dir / "reference.wav"))
        ref_btn = QPushButton("参照...")
        ref_btn.clicked.connect(self.browse_ref_wav)
        ref_layout.addWidget(self.ref_edit)
        ref_layout.addWidget(ref_btn)
        layout.addLayout(ref_layout)

        # 出力 wav
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("出力 wav:"))
        default_out = out_dir / "test_001.wav"
        self.out_edit = QLineEdit(str(default_out))
        out_btn = QPushButton("参照...")
        out_btn.clicked.connect(self.browse_out_wav)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)

        # テキスト
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

        # 初回起動の簡単な案内
        self.append_log(
            "[INFO] インストールディレクトリ配下に models/sovits, ref_wav, out を自動作成しました。\n"
            f"[INFO] SoVITS の FT モデルは: {models_s2_dir} に .pth を置いてください。\n"
            f"[INFO] 参照音声は: {ref_dir} に .wav を置いておくと、自動検出されます。"
        )

    # ---- ログ ----
    def append_log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    # ---- ブラウズ系 ----
    def browse_s2_ckpt(self):
        start_dir = str(get_install_root() / "models" / "sovits")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "SoVITS ckpt を選択",
            start_dir,
            "Checkpoint (*.pth *.ckpt);;All files (*)",
        )
        if path:
            self.s2_edit.setText(path)

    def browse_gpt_ckpt(self):
        start_dir = str(get_repo_dir())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "GPT ckpt を選択",
            start_dir,
            "Checkpoint (*.pth *.ckpt);;All files (*)",
        )
        if path:
            self.gpt_edit.setText(path)

    def browse_config(self):
        start_dir = str(get_repo_dir() / "GPT_SoVITS" / "configs")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "config (s2.json 等) を選択",
            start_dir,
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def browse_bert(self):
        start_dir = str(get_repo_dir() / "GPT_SoVITS" / "pretrained_models")
        path = QFileDialog.getExistingDirectory(self, "BERT ディレクトリを選択", start_dir)
        if path:
            self.bert_edit.setText(path)

    def browse_ref_wav(self):
        start_dir = str(get_install_root() / "ref_wav")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "参照 wav を選択",
            start_dir,
            "WAV files (*.wav);;All files (*)",
        )
        if path:
            self.ref_edit.setText(path)

    def browse_out_wav(self):
        start_dir = str(get_install_root() / "out")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "出力 wav を保存",
            start_dir,
            "WAV files (*.wav);;All files (*)",
        )
        if path:
            self.out_edit.setText(path)

    # ---- 実行 ----
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
            raise ValueError("SoVITS ckpt を指定してください。\nmodels/sovits に .pth を置くと自動認識されます。")
        s2_ckpt = Path(s2_text)
        if not s2_ckpt.is_file():
            raise ValueError(f"SoVITS ckpt ファイルが存在しません:\n{s2_ckpt}")

        ref_text = self.ref_edit.text().strip()
        if not ref_text:
            raise ValueError("参照 wav を指定してください。\nref_wav ディレクトリに .wav を置くと自動認識されます。")
        ref_wav = Path(ref_text)
        if not ref_wav.is_file():
            raise ValueError(f"参照 wav が存在しません:\n{ref_wav}")

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
