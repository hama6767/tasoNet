#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PySide6 GUI で mp3 -> dataset 生成を操作する。

追加 GUI 機能:
- 冒頭スキップ秒数 (skip_head_sec)
- 末尾スキップ秒数 (skip_tail_sec)
- 1ファイルあたりの最大セグメント数 (max_segments_per_file)
- クリップ音量の正規化 (normalize)
"""

import sys
import os
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
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QMessageBox,
    QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal

from dataset_audio_prep import VadConfig, process_mp3_folder


class WorkerThread(QThread):
    log_signal = Signal(str)
    done_signal = Signal()

    def __init__(self, input_dir, output_dir, cfg: VadConfig):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cfg = cfg

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        try:
            process_mp3_folder(
                self.input_dir,
                self.output_dir,
                self.cfg,
                progress_callback=self.log,
            )
        except Exception as e:
            self.log(f"[ERROR] {e}")
        finally:
            self.done_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Dataset Builder (mp3 -> segments)")

        central = QWidget()
        layout = QVBoxLayout(central)

        # 入力ディレクトリ
        in_layout = QHBoxLayout()
        in_label = QLabel("入力 mp3 ディレクトリ:")
        self.in_edit = QLineEdit()
        in_btn = QPushButton("参照...")
        in_btn.clicked.connect(self.browse_input)
        in_layout.addWidget(in_label)
        in_layout.addWidget(self.in_edit)
        in_layout.addWidget(in_btn)
        layout.addLayout(in_layout)

        # 出力ディレクトリ
        out_layout = QHBoxLayout()
        out_label = QLabel("出力データセットディレクトリ:")
        self.out_edit = QLineEdit()
        out_btn = QPushButton("参照...")
        out_btn.clicked.connect(self.browse_output)
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)

        # VAD 設定
        vad_layout = QHBoxLayout()
        vad_layout.addWidget(QLabel("VAD aggressiveness (0-3):"))
        self.aggr_spin = QSpinBox()
        self.aggr_spin.setRange(0, 3)
        self.aggr_spin.setValue(2)
        vad_layout.addWidget(self.aggr_spin)

        vad_layout.addWidget(QLabel("min seg (ms):"))
        self.min_seg_spin = QSpinBox()
        self.min_seg_spin.setRange(100, 20000)
        self.min_seg_spin.setValue(500)
        vad_layout.addWidget(self.min_seg_spin)

        vad_layout.addWidget(QLabel("max seg (ms):"))
        self.max_seg_spin = QSpinBox()
        self.max_seg_spin.setRange(1000, 60000)
        self.max_seg_spin.setValue(10000)
        vad_layout.addWidget(self.max_seg_spin)

        layout.addLayout(vad_layout)

        # 冒頭/末尾スキップ・最大セグメント数などの便利機能
        extra_layout = QHBoxLayout()

        extra_layout.addWidget(QLabel("冒頭スキップ (秒):"))
        self.skip_head_spin = QDoubleSpinBox()
        self.skip_head_spin.setRange(0.0, 600.0)  # 最大10分まで好きに
        self.skip_head_spin.setDecimals(1)
        self.skip_head_spin.setSingleStep(1.0)
        self.skip_head_spin.setValue(0.0)
        self.skip_head_spin.setToolTip("各 mp3 の最初のこの秒数を無視します（OP や BGM など）")
        extra_layout.addWidget(self.skip_head_spin)

        extra_layout.addWidget(QLabel("末尾スキップ (秒):"))
        self.skip_tail_spin = QDoubleSpinBox()
        self.skip_tail_spin.setRange(0.0, 600.0)
        self.skip_tail_spin.setDecimals(1)
        self.skip_tail_spin.setSingleStep(1.0)
        self.skip_tail_spin.setValue(0.0)
        self.skip_tail_spin.setToolTip("各 mp3 の最後のこの秒数を無視します（ED など）")
        extra_layout.addWidget(self.skip_tail_spin)

        extra_layout.addWidget(QLabel("最大セグメント数/ファイル:"))
        self.max_seg_count_spin = QSpinBox()
        self.max_seg_count_spin.setRange(0, 10000)
        self.max_seg_count_spin.setValue(0)  # 0 = 無制限
        self.max_seg_count_spin.setToolTip("0 の場合は無制限。大きすぎる場合に数を絞る用途など。")
        extra_layout.addWidget(self.max_seg_count_spin)

        self.normalize_checkbox = QCheckBox("クリップ音量を正規化")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("各セグメントの最大音量を揃えます（ピークを -1 dBFS に）。")
        extra_layout.addWidget(self.normalize_checkbox)

        layout.addLayout(extra_layout)

        # 実行ボタン
        self.run_btn = QPushButton("データセット生成開始")
        self.run_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.run_btn)

        # ログビュー
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setCentralWidget(central)
        self.worker = None

    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "入力 mp3 ディレクトリを選択")
        if path:
            self.in_edit.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "出力ディレクトリを選択")
        if path:
            self.out_edit.setText(path)

    def start_processing(self):
        input_dir = self.in_edit.text().strip()
        output_dir = self.out_edit.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "エラー", "有効な入力ディレクトリを指定してください。")
            return
        if not output_dir:
            QMessageBox.warning(self, "エラー", "出力ディレクトリを指定してください。")
            return

        cfg = VadConfig(
            aggressiveness=self.aggr_spin.value(),
            min_segment_ms=self.min_seg_spin.value(),
            max_segment_ms=self.max_seg_spin.value(),
            frame_ms=30,
            padding_ms=300,
            skip_head_sec=float(self.skip_head_spin.value()),
            skip_tail_sec=float(self.skip_tail_spin.value()),
            max_segments_per_file=int(self.max_seg_count_spin.value()),
            normalize=self.normalize_checkbox.isChecked(),
        )

        self.log_text.clear()
        self.run_btn.setEnabled(False)

        self.worker = WorkerThread(input_dir, output_dir, cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()
        self.append_log("処理を開始しました...\n")

    def append_log(self, msg: str):
        self.log_text.append(msg)

    def on_done(self):
        self.run_btn.setEnabled(True)
        self.append_log("すべての処理が完了しました。")
        QMessageBox.information(self, "完了", "データセット生成が完了しました。")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
