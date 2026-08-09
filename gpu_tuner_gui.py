#!/usr/bin/env python3
"""Tune exact GEMM sizes by comparing live GPU sound with a phone reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from player import Event, load_anchors, note_frequency, size_for_frequency, tuned_size


MIN_SIZE = 512
MAX_SIZE = 2048
STEP = 8


def aligned(value: int) -> int:
    return max(MIN_SIZE, min(MAX_SIZE, int(round(value / STEP) * STEP)))


class GpuTuner(QMainWindow):
    def __init__(self, labels_path: Path, tuning_path: Path, duration: float) -> None:
        super().__init__()
        self.root = Path(__file__).resolve().parent
        self.labels_path = labels_path.resolve()
        self.tuning_path = tuning_path.resolve()
        self.duration = duration
        self.anchors = load_anchors(self.labels_path)
        self.tuning = self.load_tuning()
        self.notes = self.sorted_notes()
        self.current_index = 0
        self.lower_size = MIN_SIZE
        self.upper_size = MAX_SIZE
        self.history: list[tuple[int, int, int]] = []
        self.intentional_stop = False
        self.shortcuts: list[QShortcut] = []

        self.worker = QProcess(self)
        self.worker.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.capture_output)
        self.worker.finished.connect(self.worker_finished)

        self.setWindowTitle("V100 MUSIC / REVERSE NOTE TUNER")
        self.resize(860, 680)
        self.build_ui()
        self.install_shortcuts()
        self.load_note(0)

    def load_tuning(self) -> dict:
        if self.tuning_path.exists():
            payload = json.loads(self.tuning_path.read_text(encoding="utf-8"))
        else:
            payload = {"notes": {}, "events": {}}
        payload.setdefault("notes", {})
        payload.setdefault("events", {})
        return payload

    def sorted_notes(self) -> list[str]:
        notes = list(self.tuning.get("notes", {}))
        if not notes:
            raise SystemExit("tuning.json 的 notes 为空")
        return sorted(notes, key=note_frequency)

    def current_note(self) -> str:
        return self.notes[self.current_index]

    def initial_size(self, note: str) -> int:
        frequency = note_frequency(note)
        event = Event(note, frequency, 1.0)
        size, _ = tuned_size(event, 1, self.anchors, self.tuning)
        return aligned(size)

    def build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        title = QLabel("V100 / GPU 反向音高标注")
        title.setObjectName("title")
        subtitle = QLabel("手机播放目标音；GPU 播放参考声。只判断 GPU 高了还是低了。")
        subtitle.setObjectName("subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("目标音"))
        self.note_combo = QComboBox()
        for note in self.notes:
            self.note_combo.addItem(note)
        self.note_combo.currentIndexChanged.connect(self.load_note)
        select_row.addWidget(self.note_combo, 1)
        outer.addLayout(select_row)

        target = QGroupBox("手机目标")
        target_layout = QGridLayout(target)
        self.note_label = QLabel()
        self.note_label.setObjectName("note")
        self.frequency_label = QLabel()
        self.frequency_label.setObjectName("frequency")
        target_layout.addWidget(self.note_label, 0, 0)
        target_layout.addWidget(self.frequency_label, 0, 1)
        target_layout.addWidget(QLabel("请在手机调音器/信号发生器上播放这个音名或频率"), 1, 0, 1, 2)
        outer.addWidget(target)

        gpu = QGroupBox("GPU 参考声")
        gpu_layout = QGridLayout(gpu)
        gpu_layout.addWidget(QLabel("矩阵尺寸"), 0, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(MIN_SIZE, MAX_SIZE)
        self.size_spin.setSingleStep(STEP)
        self.size_spin.setKeyboardTracking(False)
        self.size_spin.valueChanged.connect(self.size_edited)
        gpu_layout.addWidget(self.size_spin, 0, 1)
        gpu_layout.addWidget(QLabel("二分区间"), 1, 0)
        self.bounds_label = QLabel()
        gpu_layout.addWidget(self.bounds_label, 1, 1)

        self.play_button = QPushButton("播放双卡 GPU 参考声  (Space)")
        self.stop_button = QPushButton("停止  (Esc)")
        self.play_button.clicked.connect(self.play_gpu)
        self.stop_button.clicked.connect(self.stop_gpu)
        gpu_layout.addWidget(self.play_button, 2, 0)
        gpu_layout.addWidget(self.stop_button, 2, 1)
        self.auto_replay = QCheckBox("判断后自动重播新尺寸")
        self.auto_replay.setChecked(True)
        gpu_layout.addWidget(self.auto_replay, 3, 0, 1, 2)
        outer.addWidget(gpu)

        decision = QGroupBox("GPU 相对手机目标音")
        decision_layout = QGridLayout(decision)
        high = QPushButton("GPU 高了 → 增大矩阵  (H)")
        low = QPushButton("GPU 低了 → 减小矩阵  (L)")
        minus = QPushButton("矩阵 −8 / 升一点  ([)")
        plus = QPushButton("矩阵 +8 / 降一点  (])")
        match = QPushButton("匹配：保存 size 并到下一音  (M)")
        undo = QPushButton("撤销  (U)")
        reset = QPushButton("重置此音")
        high.setObjectName("high")
        low.setObjectName("low")
        match.setObjectName("match")
        high.clicked.connect(self.gpu_high)
        low.clicked.connect(self.gpu_low)
        minus.clicked.connect(lambda: self.nudge(-STEP))
        plus.clicked.connect(lambda: self.nudge(STEP))
        match.clicked.connect(self.save_and_next)
        undo.clicked.connect(self.undo)
        reset.clicked.connect(self.reset_current)
        decision_layout.addWidget(high, 0, 0)
        decision_layout.addWidget(low, 0, 1)
        decision_layout.addWidget(minus, 1, 0)
        decision_layout.addWidget(plus, 1, 1)
        decision_layout.addWidget(match, 2, 0, 1, 2)
        decision_layout.addWidget(undo, 3, 0)
        decision_layout.addWidget(reset, 3, 1)
        outer.addWidget(decision)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(300)
        outer.addWidget(QLabel("GPU 输出 / 已保存尺寸"))
        outer.addWidget(self.output, 1)

        self.statusBar().showMessage("就绪")
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111417; color: #e8ecef; font-size: 14px; }
            QLabel#title { font-size: 28px; font-weight: 800; }
            QLabel#subtitle { color: #9ca9b0; }
            QLabel#note { font-size: 38px; font-weight: 900; color: #ffd34e; }
            QLabel#frequency { font-size: 26px; font-weight: 800; color: #91d7ff; }
            QGroupBox { border: 1px solid #3b454b; margin-top: 9px; padding-top: 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #252c31; border: 1px solid #56636a; padding: 10px; }
            QPushButton:hover { background: #333d43; }
            QPushButton#high { background: #6b302d; border-color: #bd5d55; }
            QPushButton#low { background: #234f69; border-color: #4b91ba; }
            QPushButton#match { background: #315d37; border-color: #63a96c; font-weight: 800; }
            QComboBox, QSpinBox, QPlainTextEdit { background: #1b2024; border: 1px solid #465159; padding: 7px; }
            """
        )

    def install_shortcuts(self) -> None:
        bindings = {
            "Space": self.play_gpu,
            "Escape": self.stop_gpu,
            "H": self.gpu_high,
            "L": self.gpu_low,
            "[": lambda: self.nudge(-STEP),
            "]": lambda: self.nudge(STEP),
            "M": self.save_and_next,
            "U": self.undo,
        }
        for key, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def update_readout(self) -> None:
        note = self.current_note()
        self.note_label.setText(note)
        self.frequency_label.setText(f"{note_frequency(note):.2f} Hz")
        self.bounds_label.setText(f"{self.lower_size} — {self.upper_size}，步进 {STEP}")
        saved = self.tuning.get("notes", {}).get(note, {})
        self.output.setPlainText(
            "\n".join(
                f"{name:4s} -> {rule.get('size', '未定')}"
                for name, rule in self.tuning.get("notes", {}).items()
            )
            + f"\n\n当前历史步数：{len(self.history)}\n原配置：{json.dumps(saved, ensure_ascii=False)}"
        )

    def load_note(self, index: int) -> None:
        if not self.notes:
            return
        self.stop_gpu()
        self.current_index = max(0, min(index, len(self.notes) - 1))
        if self.note_combo.currentIndex() != self.current_index:
            self.note_combo.blockSignals(True)
            self.note_combo.setCurrentIndex(self.current_index)
            self.note_combo.blockSignals(False)
        self.lower_size = MIN_SIZE
        self.upper_size = MAX_SIZE
        self.history.clear()
        self.size_spin.blockSignals(True)
        self.size_spin.setValue(self.initial_size(self.current_note()))
        self.size_spin.blockSignals(False)
        self.update_readout()

    def size_edited(self, value: int) -> None:
        aligned_value = aligned(value)
        if aligned_value != value:
            self.size_spin.blockSignals(True)
            self.size_spin.setValue(aligned_value)
            self.size_spin.blockSignals(False)
        self.update_readout()

    def process_running(self) -> bool:
        return self.worker.state() != QProcess.ProcessState.NotRunning

    def stop_gpu(self) -> None:
        if self.process_running():
            self.intentional_stop = True
            self.worker.kill()
            self.worker.waitForFinished(1000)
            self.intentional_stop = False
        self.statusBar().showMessage("GPU 已停止")

    def play_gpu(self) -> None:
        self.stop_gpu()
        size = aligned(self.size_spin.value())
        command = [
            str(self.root / "probe.py"),
            "--size",
            str(size),
            "--duration",
            str(self.duration),
            "--rest",
            "0",
            "--repeats",
            "8",
            "--devices",
            "0,1",
        ]
        self.output.appendPlainText(f"\nPLAY {self.current_note()} -> {size}x{size}")
        self.statusBar().showMessage(f"双卡播放 {size}x{size}，持续 {self.duration:g}s")
        self.worker.start(sys.executable, command)

    def capture_output(self) -> None:
        text = bytes(self.worker.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
        if text:
            self.output.appendPlainText(text)

    def worker_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self.capture_output()
        if not self.intentional_stop and exit_code != 0:
            QMessageBox.warning(self, "GPU 进程失败", f"probe.py 退出码 {exit_code}\n\n{self.output.toPlainText()[-1500:]}")
        elif not self.intentional_stop:
            self.statusBar().showMessage("GPU 参考声播放完成")

    def choose_next(self, gpu_was_high: bool) -> None:
        self.stop_gpu()
        current = aligned(self.size_spin.value())
        self.history.append((self.lower_size, self.upper_size, current))
        if gpu_was_high:
            self.lower_size = min(self.upper_size, current + STEP)
        else:
            self.upper_size = max(self.lower_size, current - STEP)
        if self.lower_size > self.upper_size:
            self.lower_size = self.upper_size = current
        next_size = aligned((self.lower_size + self.upper_size) // 2)
        self.size_spin.setValue(next_size)
        self.update_readout()
        if self.auto_replay.isChecked():
            QTimer.singleShot(150, self.play_gpu)

    def gpu_high(self) -> None:
        self.choose_next(True)

    def gpu_low(self) -> None:
        self.choose_next(False)

    def nudge(self, amount: int) -> None:
        self.stop_gpu()
        current = aligned(self.size_spin.value())
        self.history.append((self.lower_size, self.upper_size, current))
        self.size_spin.setValue(aligned(current + amount))
        if self.auto_replay.isChecked():
            QTimer.singleShot(150, self.play_gpu)

    def undo(self) -> None:
        self.stop_gpu()
        if not self.history:
            self.statusBar().showMessage("没有可撤销步骤")
            return
        self.lower_size, self.upper_size, size = self.history.pop()
        self.size_spin.setValue(size)
        self.update_readout()

    def reset_current(self) -> None:
        self.stop_gpu()
        self.lower_size = MIN_SIZE
        self.upper_size = MAX_SIZE
        self.history.clear()
        self.size_spin.setValue(self.initial_size(self.current_note()))
        self.update_readout()

    def save_and_next(self) -> None:
        self.stop_gpu()
        note = self.current_note()
        size = aligned(self.size_spin.value())
        self.tuning.setdefault("notes", {})[note] = {"size": size}
        self.tuning_path.write_text(
            json.dumps(self.tuning, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.statusBar().showMessage(f"已保存 {note} -> {size}x{size}")
        next_index = (self.current_index + 1) % len(self.notes)
        QTimer.singleShot(250, lambda: self.note_combo.setCurrentIndex(next_index))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_gpu()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手机目标音 + 双卡 GPU 参考声反向标注 GUI")
    parser.add_argument("--labels", type=Path, default=Path("pitch_labels.json"))
    parser.add_argument("--tuning", type=Path, default=Path("tuning.json"))
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = GpuTuner(args.labels, args.tuning, args.duration)
    window.show()
    if args.smoke_test:
        QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
