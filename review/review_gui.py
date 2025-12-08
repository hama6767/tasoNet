#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
segments_with_text.csv を目視確認 / 編集するための GUI ツール。

想定するフォルダ構成:
    <dataset_dir>/
        segments_with_text.csv
        wavs/
          <utt_id>.wav
        ...

使い方:
    python segments_reviewer_gui.py
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QShortcut, QKeySequence
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSlider,
)


class SegmentsModel:
    """
    segments_with_text.csv を読み書きする小さなモデルクラス。
    """

    def __init__(self):
        self.dataset_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.fieldnames: List[str] = []
        self.rows: List[Dict[str, str]] = []
        self.dirty: bool = False  # 何か変更があったかどうか

    def load(self, dataset_dir: Path) -> None:
        dataset_dir = dataset_dir.resolve()
        csv_path = dataset_dir / "segments_with_text.csv"
        if not csv_path.is_file():
            # まだ transcribe していない場合は segments.csv を読んで text 列等を足す
            csv_path_alt = dataset_dir / "segments.csv"
            if not csv_path_alt.is_file():
                raise FileNotFoundError(
                    f"segments_with_text.csv も segments.csv も見つかりません: {dataset_dir}"
                )
            csv_path = csv_path_alt

        self.dataset_dir = dataset_dir
        self.csv_path = csv_path

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        # 必要な列を足しておく
        for col in ["utt_id", "relpath", "speaker", "text", "lang", "duration_sec"]:
            if col not in fieldnames:
                fieldnames.append(col)
                for r in rows:
                    r[col] = ""

        self.fieldnames = fieldnames
        self.rows = rows
        self.dirty = False

    def save(self) -> None:
        if self.dataset_dir is None or self.csv_path is None:
            return

        # バックアップを一応残す
        backup_path = self.csv_path.with_suffix(self.csv_path.suffix + ".bak")
        if self.csv_path.is_file():
            backup_path.write_text(self.csv_path.read_text(encoding="utf-8"), encoding="utf-8")

        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for r in self.rows:
                # 欠けている列があっても落ちないように
                for col in self.fieldnames:
                    r.setdefault(col, "")
                writer.writerow(r)

        self.dirty = False

    def row_count(self) -> int:
        return len(self.rows)

    def get_row(self, index: int) -> Dict[str, str]:
        return self.rows[index]

    def update_row(self, index: int, new_row: Dict[str, str]) -> None:
        self.rows[index] = new_row
        self.dirty = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Dataset Segments Reviewer")
        self.resize(1100, 700)

        self.model = SegmentsModel()
        self.current_index: int = -1

        # ==== UI 構築 ====
        central = QWidget()
        root_layout = QHBoxLayout(central)

        # 左側: リスト
        left_layout = QVBoxLayout()
        self.dataset_label = QLabel("データセット: (未ロード)")
        left_layout.addWidget(self.dataset_label)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_selection_changed)
        left_layout.addWidget(self.list_widget)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<< 前へ")
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn = QPushButton("次へ >>")
        self.next_btn.clicked.connect(self.go_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)

        self.position_label = QLabel("0 / 0")
        left_layout.addWidget(self.position_label)

        root_layout.addLayout(left_layout, 3)

        # 右側: 詳細 + プレイヤー
        right_layout = QVBoxLayout()

        # utt_id / speaker / wav path
        meta_layout = QVBoxLayout()
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("utt_id:"))
        self.utt_id_edit = QLineEdit()
        self.utt_id_edit.setReadOnly(True)
        id_layout.addWidget(self.utt_id_edit)
        meta_layout.addLayout(id_layout)

        spk_layout = QHBoxLayout()
        spk_layout.addWidget(QLabel("speaker:"))
        self.speaker_edit = QLineEdit()
        spk_layout.addWidget(self.speaker_edit)
        meta_layout.addLayout(spk_layout)

        wav_layout = QHBoxLayout()
        wav_layout.addWidget(QLabel("wav:"))
        self.wav_path_label = QLabel("(未選択)")
        self.wav_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        wav_layout.addWidget(self.wav_path_label)
        meta_layout.addLayout(wav_layout)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("duration (sec):"))
        self.duration_label = QLabel("-")
        dur_layout.addWidget(self.duration_label)
        meta_layout.addLayout(dur_layout)

        right_layout.addLayout(meta_layout)

        # Transcript 編集
        right_layout.addWidget(QLabel("transcript:"))

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("ここに文字起こしが表示されます。")
        right_layout.addWidget(self.text_edit)

        # プレイヤーコントロール
        player_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ 再生")
        self.play_btn.clicked.connect(self.play_audio)
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.clicked.connect(self.stop_audio)

        player_layout.addWidget(self.play_btn)
        player_layout.addWidget(self.stop_btn)

        player_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        player_layout.addWidget(self.volume_slider)

        right_layout.addLayout(player_layout)

        # 保存系ボタン
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_current_and_csv)
        self.save_next_btn = QPushButton("保存して次へ")
        self.save_next_btn.clicked.connect(self.save_current_and_go_next)
        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.save_next_btn)
        right_layout.addLayout(save_layout)

        # 状態表示
        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        root_layout.addLayout(right_layout, 5)

        self.setCentralWidget(central)

        # ==== メディアプレイヤー ====
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(self.volume_slider.value() / 100.0)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        # ==== メニュー ====
        self._setup_menu()

        # ==== ショートカット ====
        QShortcut(QKeySequence("Space"), self, activated=self.play_or_pause)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current_and_csv)
        QShortcut(QKeySequence("Ctrl+Right"), self, activated=self.go_next)
        QShortcut(QKeySequence("Ctrl+Left"), self, activated=self.go_prev)

    # ------------------ メニュー/アクション ------------------

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("ファイル(&F)")

        open_act = QAction("データセットを開く(&O)", self)
        open_act.triggered.connect(self.open_dataset)
        file_menu.addAction(open_act)

        save_act = QAction("CSVを保存(&S)", self)
        save_act.triggered.connect(self.save_csv_only)
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        exit_act = QAction("終了(&Q)", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    # ------------------ データセット読み込み ------------------

    def open_dataset(self):
        path_str = QFileDialog.getExistingDirectory(self, "データセットディレクトリを選択")
        if not path_str:
            return

        dataset_dir = Path(path_str)
        try:
            self.model.load(dataset_dir)
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"データセットの読み込みに失敗しました:\n{e}")
            return

        self.dataset_label.setText(f"データセット: {dataset_dir}")
        self.populate_list()
        if self.model.row_count() > 0:
            self.set_current_index(0)
        else:
            self.set_current_index(-1)

    def populate_list(self):
        self.list_widget.clear()
        for row in self.model.rows:
            utt_id = row.get("utt_id", "")
            txt = row.get("text", "").replace("\n", " ")
            if len(txt) > 40:
                txt_disp = txt[:40] + "..."
            else:
                txt_disp = txt
            item = QListWidgetItem(f"{utt_id}: {txt_disp}")
            self.list_widget.addItem(item)
        self.update_position_label()

    # ------------------ 行選択 / 表示 ------------------

    def set_current_index(self, index: int):
        if index < 0 or index >= self.model.row_count():
            self.current_index = -1
            self.list_widget.setCurrentRow(-1)
            self.clear_detail()
            return

        self.current_index = index
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(index)
        self.list_widget.blockSignals(False)

        row = self.model.get_row(index)
        self.utt_id_edit.setText(row.get("utt_id", ""))
        self.speaker_edit.setText(row.get("speaker", ""))
        self.text_edit.setPlainText(row.get("text", ""))

        relpath = row.get("relpath", "")
        if self.model.dataset_dir is not None and relpath:
            wav_path = (self.model.dataset_dir / relpath).resolve()
            self.wav_path_label.setText(str(wav_path))
        else:
            self.wav_path_label.setText("(不明)")

        dur = row.get("duration_sec") or ""
        self.duration_label.setText(dur if dur else "-")

        self.update_position_label()

    def clear_detail(self):
        self.utt_id_edit.clear()
        self.speaker_edit.clear()
        self.text_edit.clear()
        self.wav_path_label.setText("(未選択)")
        self.duration_label.setText("-")

    def on_list_selection_changed(self, row: int):
        if row == self.current_index:
            return
        # 現在行に変更があれば保存（自動保存風にするかどうかは好み）
        self.save_current_row_only()
        self.set_current_index(row)

    def update_position_label(self):
        total = self.model.row_count()
        if self.current_index < 0:
            self.position_label.setText(f"0 / {total}")
        else:
            self.position_label.setText(f"{self.current_index + 1} / {total}")

    # ------------------ ナビゲーション ------------------

    def go_prev(self):
        if self.current_index <= 0:
            return
        self.save_current_row_only()
        self.set_current_index(self.current_index - 1)

    def go_next(self):
        if self.current_index < 0:
            return
        if self.current_index >= self.model.row_count() - 1:
            return
        self.save_current_row_only()
        self.set_current_index(self.current_index + 1)

    # ------------------ プレイヤー ------------------

    def play_audio(self):
        if self.current_index < 0:
            return
        row = self.model.get_row(self.current_index)
        relpath = row.get("relpath", "")
        if not relpath or self.model.dataset_dir is None:
            QMessageBox.warning(self, "エラー", "relpath が設定されていません。")
            return
        wav_path = (self.model.dataset_dir / relpath).resolve()
        if not wav_path.is_file():
            QMessageBox.warning(self, "エラー", f"wav ファイルが見つかりません:\n{wav_path}")
            return

        url = QUrl.fromLocalFile(str(wav_path))
        self.player.setSource(url)
        self.player.play()
        self.status_label.setText(f"再生中: {wav_path}")

    def stop_audio(self):
        self.player.stop()
        self.status_label.setText("停止")

    def play_or_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.stop_audio()
        else:
            self.play_audio()

    def on_volume_changed(self, value: int):
        self.audio_output.setVolume(value / 100.0)

    # ------------------ 保存処理 ------------------

    def save_current_row_only(self):
        if self.current_index < 0:
            return
        row = self.model.get_row(self.current_index)
        row["speaker"] = self.speaker_edit.text().strip()
        row["text"] = self.text_edit.toPlainText().strip()
        # utt_id / relpath / duration_sec は変更しない前提
        self.model.update_row(self.current_index, row)
        # リスト側の表示も更新
        item = self.list_widget.item(self.current_index)
        if item is not None:
            utt_id = row.get("utt_id", "")
            txt = row.get("text", "").replace("\n", " ")
            if len(txt) > 40:
                txt_disp = txt[:40] + "..."
            else:
                txt_disp = txt
            item.setText(f"{utt_id}: {txt_disp}")

    def save_csv_only(self):
        if self.model.csv_path is None:
            return
        # 現在行も反映してから保存
        self.save_current_row_only()
        self.model.save()
        self.status_label.setText(f"CSV 保存しました: {self.model.csv_path}")

    def save_current_and_csv(self):
        self.save_csv_only()

    def save_current_and_go_next(self):
        self.save_current_row_only()
        self.model.save()
        self.go_next()

    # ------------------ 終了時確認 ------------------

    def closeEvent(self, event: QCloseEvent):
        if self.model.dirty:
            ret = QMessageBox.question(
                self,
                "確認",
                "未保存の変更があります。保存して終了しますか？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ret == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif ret == QMessageBox.StandardButton.Yes:
                try:
                    self.save_csv_only()
                except Exception as e:
                    QMessageBox.critical(self, "エラー", f"保存に失敗しました:\n{e}")
                    event.ignore()
                    return

        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
