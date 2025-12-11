#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT-SoVITS 推論専用 GUI（高速版・一括推論特化）

- GPT-SoVITS/infer_cli.py の InferenceSession を直接呼び出し、
  モデル読み込みを 1 回にまとめて高速化。
- reference ディレクトリ内の wav / txt と
  models ディレクトリ内の SoVITS ckpt 群を組み合わせて一括推論。
- reference の txt は「参照話者のスクリプト（プロンプト）」としてのみ使用し、
  生成するセリフは GUI 上のテキストボックスに書いた内容を使う。
  （1 行 1 音声。参照 × モデル × 行数 分の wav を出力）
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List

import site

# IME 関連（特に Linux + Qt6 環境向けの保険）
os.environ.setdefault("QT_IM_MODULE", "ibus")

# ユーザー site-packages を追加
user_site = os.path.expanduser(
    f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
if os.path.isdir(user_site):
    site.addsitedir(user_site)

from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices
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
    QMessageBox,
    QCheckBox,
    QPlainTextEdit,
    QProgressBar,
)

# ---- パスユーティリティ ----

def get_base_dir() -> Path:
    """
    実行ファイルのベースディレクトリを返す。
    - 生Python実行時: この .py のあるフォルダ
    - PyInstaller exe: _MEIPASS (展開先)
    """
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


BASE_DIR = get_base_dir()

# GPT-SoVITS のルート推定
def get_repo_dir() -> Path:
    base = BASE_DIR
    # exe の中では base/GPT-SoVITS に展開される想定
    candidate = base / "GPT-SoVITS"
    if candidate.exists():
        return candidate.resolve()

    # 通常のディレクトリ構成: この GUI の 1 つ上に GPT-SoVITS がある想定
    alt = base.parent / "GPT-SoVITS"
    if alt.exists():
        return alt.resolve()

    # 最後の保険（元コードの絶対パス）
    return Path("/home/hama6767/DA/tasoNet/training/GPT-SoVITS").resolve()


REPO_DIR = get_repo_dir()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# infer_cli をインポート（InferenceSession / save_wav を利用）
try:
    import infer_cli  # type: ignore
except Exception as e:
    print("[GUI] infer_cli の import に失敗しました:", e)
    infer_cli = None  # 後で GUI 側でエラーメッセージ


def get_default_bert_dir() -> Path:
    return REPO_DIR / "GPT_SoVITS" / "pretrained_models" / "bert-base-multilingual-cased"


def get_default_s2_config() -> Path:
    return REPO_DIR / "GPT_SoVITS" / "configs" / "s2.json"


# ===== 一括推論ワーカー =====

class BatchInferWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total
    done_signal = Signal(int)

    def __init__(
        self,
        ref_dir: Path,
        models_dir: Path,
        out_root: Path,
        text_list: List[str],
        gpt_ckpt: Optional[Path],
        config_path: Optional[Path],
        bert_path: Optional[Path],
        cnhubert_path: Optional[Path],
        no_half: bool,
    ):
        super().__init__()
        self.ref_dir = ref_dir
        self.models_dir = models_dir
        self.out_root = out_root
        self.text_list = text_list
        self.gpt_ckpt = gpt_ckpt
        self.config_path = config_path
        self.bert_path = bert_path
        self.cnhubert_path = cnhubert_path
        self.no_half = no_half

        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        if infer_cli is None:
            self.log("[ERROR] infer_cli モジュールがロードできていません。パス設定を確認してください。")
            self.done_signal.emit(-1)
            return

        try:
            wav_files = sorted(self.ref_dir.glob("*.wav"))
            model_files = sorted(
                [p for p in self.models_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in (".pth", ".ckpt")]
            )

            if not wav_files:
                self.log("[ERROR] reference ディレクトリに *.wav がありません。")
                self.done_signal.emit(-1)
                return
            if not model_files:
                self.log("[ERROR] models ディレクトリに .pth / .ckpt がありません。")
                self.done_signal.emit(-1)
                return
            if not self.text_list:
                self.log("[ERROR] 生成テキストが 1 行もありません。GUI で入力してください。")
                self.done_signal.emit(-1)
                return

            total_jobs = len(wav_files) * len(model_files) * len(self.text_list)
            self.log(f"[INFO] reference wav: {len(wav_files)} 個")
            self.log(f"[INFO] models      : {len(model_files)} 個")
            self.log(f"[INFO] text 行数   : {len(self.text_list)} 行")
            self.log(f"[INFO] 合計ジョブ数: {total_jobs} 件")

            # セッション初期化（最初のモデル + 設定）
            first_model = model_files[0]
            cfg_path = str(self.config_path) if self.config_path else str(get_default_s2_config())
            gpt_path = str(self.gpt_ckpt) if self.gpt_ckpt else None
            bert_path = str(self.bert_path) if self.bert_path else None
            cnhubert_path = str(self.cnhubert_path) if self.cnhubert_path else None

            self.log(f"[INFO] InferenceSession を初期化します: "
                     f"s2={first_model.name}, gpt={gpt_path}, config={cfg_path}")

            session = infer_cli.InferenceSession(
                config_path=cfg_path,
                s2_ckpt_path=str(first_model),
                gpt_ckpt_path=gpt_path,
                bert_path=bert_path,
                cnhubert_path=cnhubert_path,
                no_half=self.no_half,
            )

            overall_rc = 0
            job_idx = 0

            for model in model_files:
                if self._stop_requested:
                    break

                self.log(f"[INFO] === モデル {model.name} を処理します ===")
                if model != first_model:
                    session.change_s2_ckpt(str(model))

                model_out_root = self.out_root / model.stem
                model_out_root.mkdir(parents=True, exist_ok=True)

                for ref_wav in wav_files:
                    if self._stop_requested:
                        break

                    stem = ref_wav.stem
                    txt_path = self.ref_dir / f"{stem}.txt"

                    if not txt_path.is_file():
                        self.log(f"[WARN] {txt_path.name} がないため {ref_wav.name} をスキップします。")
                        continue

                    try:
                        prompt_text = txt_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        prompt_text = txt_path.read_text(encoding="cp932", errors="replace")

                    prompt_text = prompt_text.strip()
                    if not prompt_text:
                        self.log(f"[WARN] {txt_path.name} が空のため {ref_wav.name} をスキップします。")
                        continue

                    self.log(
                        f"[INFO] ref={ref_wav.name}, prompt={txt_path.name}, "
                        f"text 行数={len(self.text_list)}"
                    )

                    ref_out_root = model_out_root / stem
                    ref_out_root.mkdir(parents=True, exist_ok=True)

                    for line_idx, text in enumerate(self.text_list, start=1):
                        if self._stop_requested:
                            break

                        job_idx += 1
                        self.progress_signal.emit(job_idx, total_jobs)

                        out_wav = ref_out_root / f"{stem}_{line_idx:03d}.wav"

                        self.log(
                            f"[INFO] 実行: model={model.name}, ref={ref_wav.name}, "
                            f"line={line_idx}, out={out_wav.relative_to(self.out_root)}"
                        )

                        try:
                            sr, audio = session.infer(
                                ref_wav_path=str(ref_wav),
                                text=text,
                                prompt_text=prompt_text,
                            )
                            infer_cli.save_wav(sr, audio, str(out_wav))
                        except Exception as e:
                            self.log(
                                f"[ERROR] 推論中にエラー: model={model.name}, "
                                f"ref={ref_wav.name}, line={line_idx}: {e}"
                            )
                            overall_rc = -1

            if self._stop_requested:
                self.log("[INFO] ユーザー操作により一括推論を中断しました。")
                self.done_signal.emit(overall_rc or -1)
            else:
                self.done_signal.emit(overall_rc)

        except Exception:
            tb = traceback.format_exc()
            self.log("[ERROR] 一括推論中に例外が発生しました:")
            self.log(tb)
            self.done_signal.emit(-1)


# ===== GUI 本体 =====

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS Inference GUI (Batch / Fast)")
        self.resize(1100, 750)

        self.project_root = BASE_DIR.parent

        # デフォルトパス（プロジェクトルート相対）
        default_ref_dir = self.project_root / "reference"
        default_models_dir = self.project_root / "models"
        default_out_dir = self.project_root / "out"
        default_gpt_ckpt = (
            REPO_DIR
            / "GPT_SoVITS"
            / "pretrained_models"
            / "GPT-SoVITS"
            / "gsv-v2final-pretrained"
            / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        )

        central = QWidget()
        main_layout = QHBoxLayout(central)

        # 左: 設定 + ログ
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 3)

        # 右: キャラクター画像
        right_layout = QVBoxLayout()
        right_layout.addStretch()
        self.char_label = QLabel()
        self.char_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.char_label)
        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)

        layout = left_layout

        # --- パス設定 ---

        path_layout = QVBoxLayout()

        ref_dir_layout = QHBoxLayout()
        ref_dir_layout.addWidget(QLabel("reference ディレクトリ:"))
        self.ref_dir_edit = QLineEdit(str(default_ref_dir))
        ref_dir_btn = QPushButton("参照...")
        ref_dir_btn.clicked.connect(self.browse_ref_dir)
        ref_dir_layout.addWidget(self.ref_dir_edit)
        ref_dir_layout.addWidget(ref_dir_btn)
        path_layout.addLayout(ref_dir_layout)

        models_dir_layout = QHBoxLayout()
        models_dir_layout.addWidget(QLabel("models ディレクトリ (SoVITS ckpt 群):"))
        self.models_dir_edit = QLineEdit(str(default_models_dir))
        models_dir_btn = QPushButton("参照...")
        models_dir_btn.clicked.connect(self.browse_models_dir)
        models_dir_layout.addWidget(self.models_dir_edit)
        models_dir_layout.addWidget(models_dir_btn)
        path_layout.addLayout(models_dir_layout)

        out_dir_layout = QHBoxLayout()
        out_dir_layout.addWidget(QLabel("出力ルートディレクトリ:"))
        self.out_dir_edit = QLineEdit(str(default_out_dir))
        out_dir_btn = QPushButton("参照...")
        out_dir_btn.clicked.connect(self.browse_out_dir)
        out_dir_open_btn = QPushButton("開く")
        out_dir_open_btn.clicked.connect(self.open_out_dir)
        out_dir_layout.addWidget(self.out_dir_edit)
        out_dir_layout.addWidget(out_dir_btn)
        out_dir_layout.addWidget(out_dir_open_btn)
        path_layout.addLayout(out_dir_layout)

        layout.addLayout(path_layout)

        # --- モデル設定 (config / GPT / BERT / CN-HuBERT) ---

        model_layout = QVBoxLayout()

        cfg_layout = QHBoxLayout()
        cfg_layout.addWidget(QLabel("config (s2.json 等 / 任意):"))
        self.cfg_edit = QLineEdit(str(get_default_s2_config()))
        cfg_btn = QPushButton("参照...")
        cfg_btn.clicked.connect(self.browse_config)
        cfg_layout.addWidget(self.cfg_edit)
        cfg_layout.addWidget(cfg_btn)
        model_layout.addLayout(cfg_layout)

        gpt_layout = QHBoxLayout()
        gpt_layout.addWidget(QLabel("GPT ckpt (任意 / 空でデフォルト):"))
        self.gpt_edit = QLineEdit(str(default_gpt_ckpt))
        gpt_btn = QPushButton("参照...")
        gpt_btn.clicked.connect(self.browse_gpt_ckpt)
        gpt_layout.addWidget(self.gpt_edit)
        gpt_layout.addWidget(gpt_btn)
        model_layout.addLayout(gpt_layout)

        bert_layout = QHBoxLayout()
        bert_layout.addWidget(QLabel("BERT ディレクトリ or モデルID (任意):"))
        self.bert_edit = QLineEdit(str(get_default_bert_dir()))
        bert_btn = QPushButton("参照...")
        bert_btn.clicked.connect(self.browse_bert)
        bert_layout.addWidget(self.bert_edit)
        bert_layout.addWidget(bert_btn)
        model_layout.addLayout(bert_layout)

        cnhubert_layout = QHBoxLayout()
        cnhubert_layout.addWidget(QLabel("CN-HuBERT ディレクトリ (任意):"))
        self.cnhubert_edit = QLineEdit("")  # 空なら infer_cli 側でデフォルト
        cnhubert_btn = QPushButton("参照...")
        cnhubert_btn.clicked.connect(self.browse_cnhubert)
        cnhubert_layout.addWidget(self.cnhubert_edit)
        cnhubert_layout.addWidget(cnhubert_btn)
        model_layout.addLayout(cnhubert_layout)

        half_layout = QHBoxLayout()
        self.no_half_checkbox = QCheckBox("--no-half を付ける（FP32 推論）")
        self.no_half_checkbox.setChecked(True)
        half_layout.addWidget(self.no_half_checkbox)
        half_layout.addStretch()
        model_layout.addLayout(half_layout)

        layout.addLayout(model_layout)

        # --- テキスト入力 ---

        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("生成したいテキスト（1 行 1 音声。全モデル × 全reference に適用）:"))

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("例:\nあまねかなただよ、こんかなた。よろしくね。\n別のセリフ 2 行目...\n...")
        text_layout.addWidget(self.text_edit)

        text_btn_layout = QHBoxLayout()
        load_txt_btn = QPushButton("テキストファイルから読み込み")
        load_txt_btn.clicked.connect(self.load_text_from_file)
        save_txt_btn = QPushButton("テキストをファイルに保存")
        save_txt_btn.clicked.connect(self.save_text_to_file)
        text_btn_layout.addWidget(load_txt_btn)
        text_btn_layout.addWidget(save_txt_btn)
        text_layout.addLayout(text_btn_layout)

        layout.addLayout(text_layout)

        # --- 実行ボタン & 進捗 ---

        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("一括推論開始")
        self.run_btn.clicked.connect(self.start_batch_infer)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_batch_infer)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        prog_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("0 / 0")
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.progress_label)
        layout.addLayout(prog_layout)

        # --- ログ ---

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setCentralWidget(central)
        self.worker: Optional[BatchInferWorker] = None

        # キャラクター画像読み込み
        self.load_character_image()

        # infer_cli import 失敗時の警告
        if infer_cli is None:
            QMessageBox.warning(
                self,
                "エラー",
                "infer_cli.py をインポートできませんでした。\n"
                "GPT-SoVITS の配置やパス設定を確認してください。",
            )

    # ---- キャラクター画像 ----
    def load_character_image(self):
        img_path = self.project_root / "taso.png"
        if img_path.is_file():
            pix = QPixmap(str(img_path))
            if not pix.isNull():
                pix = pix.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.char_label.setPixmap(pix)
                self.char_label.setVisible(True)
                return
        self.char_label.setVisible(False)

    # ---- ブラウズ系 ----
    def browse_ref_dir(self):
        path = QFileDialog.getExistingDirectory(self, "reference ディレクトリを選択")
        if path:
            self.ref_dir_edit.setText(path)

    def browse_models_dir(self):
        path = QFileDialog.getExistingDirectory(self, "models ディレクトリを選択")
        if path:
            self.models_dir_edit.setText(path)

    def browse_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "出力ディレクトリを選択")
        if path:
            self.out_dir_edit.setText(path)

    def open_out_dir(self):
        out_path = Path(self.out_dir_edit.text().strip())
        out_path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_path.resolve())))

    def browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "config (s2.json 等) を選択",
            filter="JSON files (*.json);;All files (*)",
        )
        if path:
            self.cfg_edit.setText(path)

    def browse_gpt_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "GPT ckpt を選択",
            filter="Checkpoint (*.pth *.ckpt);;All files (*)",
        )
        if path:
            self.gpt_edit.setText(path)

    def browse_bert(self):
        path = QFileDialog.getExistingDirectory(self, "BERT ディレクトリを選択")
        if path:
            self.bert_edit.setText(path)

    def browse_cnhubert(self):
        path = QFileDialog.getExistingDirectory(self, "CN-HuBERT ディレクトリを選択")
        if path:
            self.cnhubert_edit.setText(path)

    # ---- テキストファイル入出力 ----

    def load_text_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "生成テキストを読み込む",
            filter="Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            txt = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = Path(path).read_text(encoding="cp932", errors="replace")
        self.text_edit.setPlainText(txt)

    def save_text_to_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "生成テキストを保存",
            filter="Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        text = self.text_edit.toPlainText()
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "保存エラー", f"テキストの保存に失敗しました: {e}")

    # ---- ログ & 進捗 ----

    def append_log(self, msg: str):
        self.log_text.appendPlainText(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_progress(self, current: int, total: int):
        if total <= 0:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_label.setText("0 / 0")
            return

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current} / {total}")

    def on_done(self, rc: int):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        if rc == 0:
            self.append_log("[INFO] 一括推論が正常に終了しました。")
            QMessageBox.information(self, "完了", "一括推論が完了しました。")
        else:
            self.append_log(f"[ERROR] 一括推論がエラー終了しました (rc={rc})。")
            QMessageBox.warning(self, "エラー", f"一括推論がエラー終了しました (rc={rc})。")

    # ---- 設定構築 ----

    def build_batch_cfg_from_ui(self):
        ref_dir_text = self.ref_dir_edit.text().strip()
        if not ref_dir_text:
            raise ValueError("reference ディレクトリを指定してください。")
        ref_dir = Path(ref_dir_text)
        if not ref_dir.is_dir():
            raise ValueError("reference ディレクトリが存在しません。")

        models_dir_text = self.models_dir_edit.text().strip()
        if not models_dir_text:
            raise ValueError("models ディレクトリを指定してください。")
        models_dir = Path(models_dir_text)
        if not models_dir.is_dir():
            raise ValueError("models ディレクトリが存在しません。")

        out_root_text = self.out_dir_edit.text().strip()
        if not out_root_text:
            raise ValueError("出力ディレクトリを指定してください。")
        out_root = Path(out_root_text)
        out_root.mkdir(parents=True, exist_ok=True)

        raw_text = self.text_edit.toPlainText()
        text_list = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not text_list:
            raise ValueError("生成テキストを 1 行以上入力してください。")

        gpt_text = self.gpt_edit.text().strip()
        gpt_ckpt = Path(gpt_text) if gpt_text else None

        cfg_text = self.cfg_edit.text().strip()
        config_path = Path(cfg_text) if cfg_text else None

        bert_text = self.bert_edit.text().strip()
        bert_path = Path(bert_text) if bert_text else None

        cnhubert_text = self.cnhubert_edit.text().strip()
        cnhubert_path = Path(cnhubert_text) if cnhubert_text else None

        no_half = self.no_half_checkbox.isChecked()

        return dict(
            ref_dir=ref_dir,
            models_dir=models_dir,
            out_root=out_root,
            text_list=text_list,
            gpt_ckpt=gpt_ckpt,
            config_path=config_path,
            bert_path=bert_path,
            cnhubert_path=cnhubert_path,
            no_half=no_half,
        )

    # ---- 実行 ----

    def start_batch_infer(self):
        try:
            cfg = self.build_batch_cfg_from_ui()
        except ValueError as e:
            QMessageBox.warning(self, "入力エラー", str(e))
            return

        self.log_text.clear()
        self.append_log("[INFO] 一括推論を開始します...")
        self.update_progress(0, 0)

        self.run_btn.setEnabled(True)  # 直後に False にするが、意図的に明示
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = BatchInferWorker(**cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()

    def stop_batch_infer(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.append_log("[INFO] 停止要求を送りました。現在のジョブが終了するまで少し待ちます。")
            self.stop_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
