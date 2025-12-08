#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper 文字起こしを Docker コンテナ内で実行する PySide6 GUI。

前提:
- Docker イメージ: tts-whisper-gpu
- transcribe_segments.py がこのファイルと同じディレクトリにある
"""

import sys
import subprocess
from pathlib import Path

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
    QComboBox,
)
from PySide6.QtCore import QThread, Signal


class DockerWorkerThread(QThread):
    log_signal = Signal(str)
    done_signal = Signal(int)

    def __init__(
        self,
        image_name: str,
        dataset_dir_host: Path,
        model_name: str,
        use_cpu: bool,
        language: str,
        speaker: str,
        overwrite: bool,
    ):
        super().__init__()
        self.image_name = image_name
        self.dataset_dir_host = dataset_dir_host
        self.model_name = model_name
        self.use_cpu = use_cpu
        self.language = language
        self.speaker = speaker
        self.overwrite = overwrite

    def run(self):
        # プロジェクトルート: transcribe_docker_gui.py のあるディレクトリを app_root とする
        app_root = Path(__file__).resolve().parent

        dataset_dir_host = self.dataset_dir_host.resolve()

        # コンテナ内のパス
        dataset_dir_container = Path("/workspace/dataset")
        app_root_container = Path("/workspace/app")

        # Docker コマンドを組み立て（shell=False 前提）
        cmd = [
            "docker", "run", "--rm",
        ]

        if not self.use_cpu:
            # GPU 使用
            cmd += ["--gpus", "all"]

        # devcontainer と合わせておく（大きめのモデルでOOMしにくくする）
        cmd += ["--shm-size", "8g"]

        # ボリュームマウント
        cmd += [
            "-v",
            f"{str(dataset_dir_host)}:{str(dataset_dir_container)}",
            "-v",
            f"{str(app_root)}:{str(app_root_container)}",
            "-w",
            str(app_root_container),
            self.image_name,
            "python",
            "transcribe_segments.py",
            str(dataset_dir_container),
            "--model",
            self.model_name,
            "--device",
            "cpu" if self.use_cpu else "cuda",
            "--language",
            self.language,
            "--speaker",
            self.speaker,
        ]

        if self.overwrite:
            cmd.append("--overwrite")

        self.log_signal.emit("実行コマンド: " + " ".join(cmd))

        try:
            # stdout+stderr をまとめて読みながらログに流す
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_signal.emit(line.rstrip("\n"))

            proc.wait()
            rc = proc.returncode
        except FileNotFoundError:
            self.log_signal.emit("[ERROR] docker コマンドが見つかりません。Docker がインストールされているか確認してください。")
            rc = -1
        except Exception as e:
            self.log_signal.emit(f"[ERROR] 予期しないエラー: {e}")
            rc = -1

        self.done_signal.emit(rc)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Dataset Transcriber (Whisper in Docker)")

        central = QWidget()
        layout = QVBoxLayout(central)

        # データセットディレクトリ
        ds_layout = QHBoxLayout()
        ds_label = QLabel("データセットディレクトリ (segments.csv がある場所):")
        self.dataset_edit = QLineEdit()
        ds_btn = QPushButton("参照...")
        ds_btn.clicked.connect(self.browse_dataset)
        ds_layout.addWidget(ds_label)
        ds_layout.addWidget(self.dataset_edit)
        ds_layout.addWidget(ds_btn)
        layout.addLayout(ds_layout)

        # Docker イメージ名
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Docker イメージ名:"))
        self.image_edit = QLineEdit("tts-whisper-gpu")
        img_layout.addWidget(self.image_edit)
        layout.addLayout(img_layout)

        # モデル / デバイス
        mdl_layout = QHBoxLayout()
        mdl_layout.addWidget(QLabel("Whisperモデル:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        )
        idx = self.model_combo.findText("large-v3")
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        mdl_layout.addWidget(self.model_combo)

        self.cpu_checkbox = QCheckBox("CPUで実行（遅い）")
        self.cpu_checkbox.setToolTip("チェックを入れると --gpus を付けず device=cpu で実行します。")
        mdl_layout.addWidget(self.cpu_checkbox)

        layout.addLayout(mdl_layout)

        # 言語 / speaker / overwrite
        misc_layout = QHBoxLayout()
        misc_layout.addWidget(QLabel("言語コード:"))
        self.lang_edit = QLineEdit("ja")
        self.lang_edit.setFixedWidth(80)
        misc_layout.addWidget(self.lang_edit)

        misc_layout.addWidget(QLabel("speaker ID:"))
        self.speaker_edit = QLineEdit("spk")
        misc_layout.addWidget(self.speaker_edit)

        self.overwrite_checkbox = QCheckBox("既存のtextを上書き")
        self.overwrite_checkbox.setChecked(False)
        misc_layout.addWidget(self.overwrite_checkbox)

        layout.addLayout(misc_layout)

        # 実行ボタン
        self.run_btn = QPushButton("Docker で文字起こし開始")
        self.run_btn.clicked.connect(self.start_transcribe)
        layout.addWidget(self.run_btn)

        # ログ
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setCentralWidget(central)
        self.worker = None

    def browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "データセットディレクトリを選択")
        if path:
            self.dataset_edit.setText(path)

    def start_transcribe(self):
        dataset_dir_text = self.dataset_edit.text().strip()
        if not dataset_dir_text:
            QMessageBox.warning(self, "エラー", "データセットディレクトリを指定してください。")
            return

        dataset_dir = Path(dataset_dir_text)
        if not dataset_dir.is_dir():
            QMessageBox.warning(self, "エラー", "データセットディレクトリが存在しません。")
            return

        segments_csv = dataset_dir / "segments.csv"
        if not segments_csv.is_file():
            QMessageBox.warning(
                self,
                "エラー",
                f"segments.csv が見つかりません: {segments_csv}",
            )
            return

        image_name = self.image_edit.text().strip() or "tts-whisper-gpu"
        model_name = self.model_combo.currentText()
        use_cpu = self.cpu_checkbox.isChecked()
        language = self.lang_edit.text().strip() or "ja"
        speaker = self.speaker_edit.text().strip() or "spk"
        overwrite = self.overwrite_checkbox.isChecked()

        self.log_text.clear()
        self.run_btn.setEnabled(False)

        self.worker = DockerWorkerThread(
            image_name=image_name,
            dataset_dir_host=dataset_dir,
            model_name=model_name,
            use_cpu=use_cpu,
            language=language,
            speaker=speaker,
            overwrite=overwrite,
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()

        self.append_log("Docker で文字起こしを開始しました...\n")

    def append_log(self, msg: str):
        self.log_text.append(msg)

    def on_done(self, rc: int):
        self.run_btn.setEnabled(True)
        if rc == 0:
            self.append_log("すべての文字起こしが完了しました。")
            QMessageBox.information(self, "完了", "segments_with_text.csv の生成が完了しました。")
        else:
            self.append_log(f"文字起こしプロセスが異常終了しました (return code={rc})。")
            QMessageBox.warning(self, "エラー", f"Docker 実行がエラー終了しました (rc={rc})。")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
