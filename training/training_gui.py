#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ホスト側で実行する GPT-SoVITS 学習 GUI。

前提:
  - Docker イメージ: gpt-sovits-train
  - イメージ内には:
      /workspace/GPT-SoVITS
      /workspace/gpt_sovits_train_inner.py
    が存在する
  - ホスト側には PySide6 だけあればよい

ワークスペースルート配下に:
  - データセット (wav)
  - gpt_sovits_*.list
  - s2/s1 config
  - 事前学習 ckpt
などを置く想定で、`-v root:root` で丸ごとマウントします。
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

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


class DockerTrainWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal(int)

    def __init__(
        self,
        image_name: str,
        workspace_root: Path,
        list_path: Path,
        exp_name: str,
        output_root: Path,
        gpus: str,
        half: bool,
        s2_config: Optional[Path],
        s1_config: Optional[Path],
        bert_dir: Optional[Path],
        cnhubert_dir: Optional[Path],
        semantic_s2_ckpt: Optional[Path],
        semantic_s2_config: Optional[Path],
        stages: str,
    ):
        super().__init__()
        self.image_name = image_name
        self.workspace_root = workspace_root
        self.list_path = list_path
        self.exp_name = exp_name
        self.output_root = output_root
        self.gpus = gpus
        self.half = half
        self.s2_config = s2_config
        self.s1_config = s1_config
        self.bert_dir = bert_dir
        self.cnhubert_dir = cnhubert_dir
        self.semantic_s2_ckpt = semantic_s2_ckpt
        self.semantic_s2_config = semantic_s2_config
        self.stages = stages

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def run(self):
        # docker run コマンドを組み立てる
        ws = self.workspace_root.resolve()
        list_path = self.list_path.resolve()
        out_root = self.output_root.resolve()

        # コンテナ内ではホストと同じ絶対パスで見えるようにする (-v root:root)
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--shm-size",
            "16g",
            "-v",
            f"{str(ws)}:{str(ws)}",
            self.image_name,
            "python",
            "/workspace/gpt_sovits_train_inner.py",
            "--list",
            str(list_path),
            "--exp-name",
            self.exp_name,
            "--output-root",
            str(out_root),
            "--gpus",
            self.gpus,
            "--stages",
            self.stages,
        ]

        if self.half:
            cmd.append("--half")
        if self.s2_config is not None:
            cmd += ["--s2-config", str(self.s2_config.resolve())]
        if self.s1_config is not None:
            cmd += ["--s1-config", str(self.s1_config.resolve())]
        if self.bert_dir is not None:
            cmd += ["--bert-dir", str(self.bert_dir.resolve())]
        if self.cnhubert_dir is not None:
            cmd += ["--cnhubert-dir", str(self.cnhubert_dir.resolve())]
        if self.semantic_s2_ckpt is not None:
            cmd += ["--semantic-s2-ckpt", str(self.semantic_s2_ckpt.resolve())]
        if self.semantic_s2_config is not None:
            cmd += ["--semantic-s2-config", str(self.semantic_s2_config.resolve())]

        self.log("実行コマンド:")
        self.log(" ".join(cmd))

        rc = 0
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log(line.rstrip("\n"))
            proc.wait()
            rc = proc.returncode
        except FileNotFoundError:
            self.log("[ERROR] docker コマンドが見つかりません。Docker がインストールされているか確認してください。")
            rc = -1
        except Exception as e:
            self.log(f"[ERROR] 予期しないエラー: {e}")
            rc = -1

        self.done_signal.emit(rc)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS Trainer (Docker)")
        self.resize(1000, 700)

        central = QWidget()
        layout = QVBoxLayout(central)

        # Docker イメージ名
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("Docker イメージ名:"))
        self.image_edit = QLineEdit("gpt-sovits-train")
        img_layout.addWidget(self.image_edit)
        layout.addLayout(img_layout)

        # ワークスペースルート
        ws_layout = QHBoxLayout()
        ws_layout.addWidget(QLabel("ワークスペースルート:"))
        # ★修正: 画像の値に合わせる
        self.ws_edit = QLineEdit("/home/hama6767/DA/tasoNet")
        ws_btn = QPushButton("参照...")
        ws_btn.clicked.connect(self.browse_workspace)
        ws_layout.addWidget(self.ws_edit)
        ws_layout.addWidget(ws_btn)
        layout.addLayout(ws_layout)

        # .list
        list_layout = QHBoxLayout()
        list_layout.addWidget(QLabel(".list ファイル (train):"))
        # ★修正: 画像の値に合わせる
        self.list_edit = QLineEdit(
            "/home/hama6767/DA/tasoNet/converter/gpt_sovits_streamer.list"
        )
        list_btn = QPushButton("参照...")
        list_btn.clicked.connect(self.browse_list)
        list_layout.addWidget(self.list_edit)
        list_layout.addWidget(list_btn)
        layout.addLayout(list_layout)

        # exp_name / output_root
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("exp_name:"))
        self.exp_edit = QLineEdit("exp_streamer")
        exp_layout.addWidget(self.exp_edit)

        exp_layout.addWidget(QLabel("output_root:"))
        # ★修正: 画像の値に合わせる (元のコードもこれでしたが念のため確認)
        self.out_edit = QLineEdit("/home/hama6767/DA/tasoNet")
        out_btn = QPushButton("参照...")
        out_btn.clicked.connect(self.browse_output_root)
        exp_layout.addWidget(self.out_edit)
        exp_layout.addWidget(out_btn)
        layout.addLayout(exp_layout)

        # GPU / half
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("GPU IDs (例: 0 or 0,1):"))
        self.gpu_edit = QLineEdit("0")
        gpu_layout.addWidget(self.gpu_edit)

        self.half_checkbox = QCheckBox("特徴抽出を half 精度で実行")
        self.half_checkbox.setChecked(True)
        gpu_layout.addWidget(self.half_checkbox)

        layout.addLayout(gpu_layout)

        # 1a/1b/1c 用 BERT / CNHubert / semantic
        feat_layout = QVBoxLayout()

        bert_layout = QHBoxLayout()
        bert_layout.addWidget(QLabel("BERT ディレクトリ (任意):"))
        self.bert_edit = QLineEdit("")
        bert_btn = QPushButton("参照...")
        bert_btn.clicked.connect(self.browse_bert)
        bert_layout.addWidget(self.bert_edit)
        bert_layout.addWidget(bert_btn)
        feat_layout.addLayout(bert_layout)

        hubert_layout = QHBoxLayout()
        hubert_layout.addWidget(QLabel("CNHubert ベースディレクトリ (任意):"))
        # 画像の値 (Docker内のパス)
        self.hubert_edit = QLineEdit(
            "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/chinese-hubert-base"
        )
        hubert_btn = QPushButton("参照...")
        hubert_btn.clicked.connect(self.browse_hubert)
        hubert_layout.addWidget(self.hubert_edit)
        hubert_layout.addWidget(hubert_btn)
        feat_layout.addLayout(hubert_layout)

        sem_s2_layout = QHBoxLayout()
        sem_s2_layout.addWidget(QLabel("3-get-semantic 用 SoVITS ckpt (任意):"))
        # ★修正: 画像の値に合わせる
        self.semantic_ckpt_edit = QLineEdit(
            "/home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/pretrained_models/GPT-SoVITS/s2G488k.pth"
        )
        sem_s2_btn = QPushButton("参照...")
        sem_s2_btn.clicked.connect(self.browse_semantic_ckpt)
        sem_s2_layout.addWidget(self.semantic_ckpt_edit)
        sem_s2_layout.addWidget(sem_s2_btn)
        feat_layout.addLayout(sem_s2_layout)

        sem_cfg_layout = QHBoxLayout()
        sem_cfg_layout.addWidget(QLabel("3-get-semantic 用 SoVITS config (任意):"))
        # ★修正: 画像の値に合わせる (ファイル名が s2_semantic.json)
        self.semantic_cfg_edit = QLineEdit(
            "/home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/configs/s2_semantic.json"
        )
        sem_cfg_btn = QPushButton("参照...")
        sem_cfg_btn.clicked.connect(self.browse_semantic_cfg)
        sem_cfg_layout.addWidget(self.semantic_cfg_edit)
        sem_cfg_layout.addWidget(sem_cfg_btn)
        feat_layout.addLayout(sem_cfg_layout)

        layout.addLayout(feat_layout)

        # S2/S1 config + FT 用
        cfg_layout = QVBoxLayout()

        s2_layout = QHBoxLayout()
        s2_layout.addWidget(QLabel("SoVITS config (s2.json):"))
        # ★修正: 画像の値に合わせる
        self.s2_edit = QLineEdit(
            "/home/hama6767/DA/tasoNet/training/GPT-SoVITS/GPT_SoVITS/configs/s2.json"
        )
        s2_btn = QPushButton("参照...")
        s2_btn.clicked.connect(self.browse_s2_config)
        s2_layout.addWidget(self.s2_edit)
        s2_layout.addWidget(s2_btn)
        cfg_layout.addLayout(s2_layout)

        pre_s2_layout = QHBoxLayout()
        pre_s2_layout.addWidget(QLabel("FT 用 SoVITS 事前学習 ckpt (pretrained_s2G に反映):"))
        self.pre_s2_edit = QLineEdit("")
        pre_s2_btn = QPushButton("参照...")
        pre_s2_btn.clicked.connect(self.browse_pre_s2)
        pre_s2_layout.addWidget(self.pre_s2_edit)
        pre_s2_layout.addWidget(pre_s2_btn)
        cfg_layout.addLayout(pre_s2_layout)

        s1_layout = QHBoxLayout()
        s1_layout.addWidget(QLabel("GPT(Stage1) config (s1longer-v2.yaml 等, 任意):"))
        # 画像の値 (Docker内のパス)
        self.s1_edit = QLineEdit(
            "/workspace/GPT-SoVITS/GPT_SoVITS/configs/s1longer-v2.yaml"
        )
        s1_btn = QPushButton("参照...")
        s1_btn.clicked.connect(self.browse_s1_config)
        s1_layout.addWidget(self.s1_edit)
        s1_layout.addWidget(s1_btn)
        cfg_layout.addLayout(s1_layout)

        layout.addLayout(cfg_layout)

        # ステージ選択
        stage_layout = QHBoxLayout()
        self.chk_1a = QCheckBox("1a")
        self.chk_1a.setChecked(True)
        self.chk_1b = QCheckBox("1b")
        self.chk_1b.setChecked(True)
        self.chk_1c = QCheckBox("1c")
        self.chk_1c.setChecked(True)
        self.chk_s2 = QCheckBox("S2")
        self.chk_s2.setChecked(True)
        self.chk_s1 = QCheckBox("S1")
        # ★修正: 画像では S1 にチェックが入っているため True に変更
        self.chk_s1.setChecked(True)

        for w in (self.chk_1a, self.chk_1b, self.chk_1c, self.chk_s2, self.chk_s1):
            stage_layout.addWidget(w)

        layout.addLayout(stage_layout)

        # 実行ボタン
        self.run_btn = QPushButton("Docker で学習開始")
        self.run_btn.clicked.connect(self.start_training)
        layout.addWidget(self.run_btn)

        # ログ
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setCentralWidget(central)

        self.worker: Optional[DockerTrainWorker] = None

    # ---- ブラウズ系 ----

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "ワークスペースルートを選択")
        if path:
            self.ws_edit.setText(path)

    def browse_list(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            ".list ファイルを選択",
            filter="List files (*.list);;All files (*)",
        )
        if path:
            self.list_edit.setText(path)

    def browse_output_root(self):
        path = QFileDialog.getExistingDirectory(self, "output_root を選択")
        if path:
            self.out_edit.setText(path)

    def browse_bert(self):
        path = QFileDialog.getExistingDirectory(self, "BERT ディレクトリを選択")
        if path:
            self.bert_edit.setText(path)

    def browse_hubert(self):
        path = QFileDialog.getExistingDirectory(self, "CNHubert ベースディレクトリを選択")
        if path:
            self.hubert_edit.setText(path)

    def browse_semantic_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "3-get-semantic 用 SoVITS ckpt を選択",
            filter="Checkpoint (*.ckpt *.pth);;All files (*)",
        )
        if path:
            self.semantic_ckpt_edit.setText(path)

    def browse_semantic_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "3-get-semantic 用 SoVITS config を選択",
            filter="JSON files (*.json);;All files (*)",
        )
        if path:
            self.semantic_cfg_edit.setText(path)

    def browse_s2_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "s2.json を選択",
            filter="JSON files (*.json);;All files (*)",
        )
        if path:
            self.s2_edit.setText(path)

    def browse_pre_s2(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "FT 用 SoVITS ckpt を選択",
            filter="Checkpoint (*.ckpt *.pth);;All files (*)",
        )
        if path:
            self.pre_s2_edit.setText(path)

    def browse_s1_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "s1 config (yaml) を選択",
            filter="YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.s1_edit.setText(path)

    # ---- S2 config の FT パッチ ----

    def patch_s2_config_if_needed(self, s2_path: Path, pre_s2_path: Optional[Path], exp_name: str) -> Path:
        """
        pretrained_s2G を上書きした s2 config を <stem>_<exp>_ft.json として作る
        """
        if pre_s2_path is None:
            return s2_path

        s2_path = s2_path.resolve()
        pre_s2_path = pre_s2_path.resolve()
        with s2_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "train" not in data:
            data["train"] = {}
        data["train"]["pretrained_s2G"] = str(pre_s2_path)

        out_path = s2_path.with_name(f"{s2_path.stem}_{exp_name}_ft.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.append_log(f"[INFO] s2 config を FT 用にパッチしました: {out_path}")
        return out_path

    # ---- 実行 ----

    def start_training(self):
        try:
            cfg = self.build_config_from_ui()
        except ValueError as e:
            QMessageBox.warning(self, "入力エラー", str(e))
            return

        self.log_text.clear()
        self.append_log("[INFO] Docker で学習パイプラインを開始します...")
        self.run_btn.setEnabled(False)

        self.worker = DockerTrainWorker(**cfg)
        self.worker.log_signal.connect(self.append_log)
        self.worker.done_signal.connect(self.on_done)
        self.worker.start()

    def build_config_from_ui(self):
        image_name = self.image_edit.text().strip() or "gpt-sovits-train"

        ws_text = self.ws_edit.text().strip()
        if not ws_text:
            raise ValueError("ワークスペースルートを指定してください。")
        ws = Path(ws_text).resolve()
        if not ws.is_dir():
            raise ValueError("ワークスペースルートが存在しません。")

        list_text = self.list_edit.text().strip()
        if not list_text:
            raise ValueError(".list ファイルを指定してください。")
        list_path = Path(list_text).resolve()
        if not list_path.is_file():
            raise ValueError(".list ファイルが存在しません。")

        # .list や config は ws 配下にあることを推奨
        if ws not in list_path.parents:
            raise ValueError(".list はワークスペースルート配下に置いてください。")

        exp_name = self.exp_edit.text().strip()
        if not exp_name:
            raise ValueError("exp_name を入力してください。")

        out_text = self.out_edit.text().strip()
        if out_text:
            out_root = Path(out_text).resolve()
        else:
            out_root = ws / "gpt_sovits_runs" / exp_name
        out_root.parent.mkdir(parents=True, exist_ok=True)

        gpus = self.gpu_edit.text().strip() or "0"
        half = self.half_checkbox.isChecked()

        # 1a/1b/1c 用
        bert_dir = Path(self.bert_edit.text().strip()).resolve() if self.bert_edit.text().strip() else None
        hubert_dir = Path(self.hubert_edit.text().strip()).resolve() if self.hubert_edit.text().strip() else None
        semantic_ckpt = Path(self.semantic_ckpt_edit.text().strip()).resolve() if self.semantic_ckpt_edit.text().strip() else None
        semantic_cfg = Path(self.semantic_cfg_edit.text().strip()).resolve() if self.semantic_cfg_edit.text().strip() else None

        # S2/S1 config
        s2_config = Path(self.s2_edit.text().strip()).resolve() if self.s2_edit.text().strip() else None
        if self.chk_s2.isChecked() and s2_config is None:
            raise ValueError("S2 を実行するには s2.json を指定してください。")

        pre_s2 = Path(self.pre_s2_edit.text().strip()).resolve() if self.pre_s2_edit.text().strip() else None
        # s2.json を FT 用にパッチする
        if s2_config is not None:
            s2_config = self.patch_s2_config_if_needed(s2_config, pre_s2, exp_name)

        s1_config = Path(self.s1_edit.text().strip()).resolve() if self.s1_edit.text().strip() else None
        if self.chk_s1.isChecked() and s1_config is None:
            raise ValueError("S1 を実行するには s1 config を指定してください。")

        stages = []
        if self.chk_1a.isChecked():
            stages.append("1a")
        if self.chk_1b.isChecked():
            stages.append("1b")
        if self.chk_1c.isChecked():
            stages.append("1c")
        if self.chk_s2.isChecked():
            stages.append("s2")
        if self.chk_s1.isChecked():
            stages.append("s1")
        if not stages:
            raise ValueError("少なくとも1つはステージを選択してください。")

        stages_str = ",".join(stages)

        cfg = dict(
            image_name=image_name,
            workspace_root=ws,
            list_path=list_path,
            exp_name=exp_name,
            output_root=out_root,
            gpus=gpus,
            half=half,
            s2_config=s2_config,
            s1_config=s1_config,
            bert_dir=bert_dir,
            cnhubert_dir=hubert_dir,
            semantic_s2_ckpt=semantic_ckpt,
            semantic_s2_config=semantic_cfg,
            stages=stages_str,
        )
        return cfg

    def append_log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def on_done(self, rc: int):
        self.run_btn.setEnabled(True)
        if rc == 0:
            self.append_log("[INFO] 学習パイプラインが正常に終了しました。")
            QMessageBox.information(self, "完了", "学習パイプラインが完了しました。")
        else:
            self.append_log(f"[ERROR] Docker プロセスが異常終了しました (rc={rc})。")
            QMessageBox.warning(self, "エラー", f"Docker 実行がエラー終了しました (rc={rc})。")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()