#!/usr/bin/env python3
"""Interactive PyQt6 pitch calibration by logarithmic frequency bisection."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile
from scipy.signal import resample_poly


MIN_FREQUENCY = 40.0
INITIAL_FREQUENCY = 20_000.0
MAX_FREQUENCY = 22_000.0
REFERENCE_SECONDS = 1.5


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 10**9, path.name)


def discover_files(explicit: list[Path]) -> list[Path]:
    if explicit:
        files = explicit
    else:
        denoised = Path("denoised")
        files = list(denoised.glob("*_denoised.wav")) if denoised.exists() else []
        if not files:
            files = [path for path in Path.cwd().glob("*.wav") if path.stem[0:1].isdigit()]
    return sorted((path.resolve() for path in files if path.exists()), key=numeric_key)


class PitchAnnotator(QMainWindow):
    def __init__(
        self,
        files: list[Path],
        output_path: Path,
        start_seconds: float,
        sample_seconds: float,
    ) -> None:
        super().__init__()
        self.files = files
        self.output_path = output_path.resolve()
        self.results = self.load_results()
        self.current_index = 0
        self.low_hz = MIN_FREQUENCY
        self.high_hz = MAX_FREQUENCY
        self.reference_hz = INITIAL_FREQUENCY
        self.history: list[tuple[float, float, float]] = []
        self.sequence: list[str] = []
        self.sequence_active = False
        self.shortcuts: list[QShortcut] = []
        self.audio_sink: QAudioSink | None = None
        self.audio_buffer: QBuffer | None = None
        self.audio_bytes: QByteArray | None = None
        self.audio_devices = list(QMediaDevices.audioOutputs())

        self.setWindowTitle("V100 MUSIC / PITCH BISECTION")
        self.resize(900, 720)
        self.build_ui(start_seconds, sample_seconds)
        self.install_shortcuts()
        self.load_target(0)

    def build_ui(self, start_seconds: float, sample_seconds: float) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        title = QLabel("V100 / FP32 音高二分标注器")
        title.setObjectName("title")
        subtitle = QLabel("先听参考正弦，再听 GPU 录音。只判断参考音：高了，还是低了。")
        subtitle.setObjectName("subtitle")
        warning = QLabel("注意：20000 Hz 常被耳朵、扬声器或声卡低通滤掉；听不到不等于播放失败。")
        warning.setObjectName("warning")
        outer.addWidget(title)
        outer.addWidget(subtitle)
        outer.addWidget(warning)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("录音"))
        self.file_combo = QComboBox()
        for path in self.files:
            self.file_combo.addItem(path.name, path)
        self.file_combo.currentIndexChanged.connect(self.load_target)
        file_row.addWidget(self.file_combo, 1)
        open_button = QPushButton("添加 WAV…")
        open_button.clicked.connect(self.add_files)
        file_row.addWidget(open_button)
        outer.addLayout(file_row)

        readout = QGroupBox("当前比较")
        readout_layout = QGridLayout(readout)
        self.frequency_label = QLabel()
        self.frequency_label.setObjectName("frequency")
        self.bounds_label = QLabel()
        self.iteration_label = QLabel()
        readout_layout.addWidget(QLabel("参考正弦"), 0, 0)
        readout_layout.addWidget(self.frequency_label, 0, 1)
        readout_layout.addWidget(QLabel("搜索区间"), 1, 0)
        readout_layout.addWidget(self.bounds_label, 1, 1)
        readout_layout.addWidget(QLabel("已判断"), 2, 0)
        readout_layout.addWidget(self.iteration_label, 2, 1)
        outer.addWidget(readout)

        playback = QGroupBox("试听")
        playback_layout = QGridLayout(playback)
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("音频输出"))
        self.device_combo = QComboBox()
        default_device = QMediaDevices.defaultAudioOutput()
        default_index = 0
        for index, device in enumerate(self.audio_devices):
            self.device_combo.addItem(device.description(), device)
            if device == default_device:
                default_index = index
        self.device_combo.setCurrentIndex(default_index)
        self.device_combo.currentIndexChanged.connect(self.audio_device_changed)
        device_row.addWidget(self.device_combo, 1)
        playback_layout.addLayout(device_row, 0, 0, 1, 2)

        self.reference_button = QPushButton("播放参考音  (R)")
        self.recording_button = QPushButton("播放 GPU 录音  (G)")
        self.ab_button = QPushButton("播放 A/B  (Space)")
        self.stop_button = QPushButton("停止  (Esc)")
        self.test_button = QPushButton("播放 1000 Hz 设备测试")
        self.reference_button.clicked.connect(self.play_reference)
        self.recording_button.clicked.connect(self.play_recording)
        self.ab_button.clicked.connect(self.play_ab)
        self.stop_button.clicked.connect(self.stop_audio)
        self.test_button.clicked.connect(self.play_device_test)
        playback_layout.addWidget(self.reference_button, 1, 0)
        playback_layout.addWidget(self.recording_button, 1, 1)
        playback_layout.addWidget(self.ab_button, 2, 0)
        playback_layout.addWidget(self.stop_button, 2, 1)
        playback_layout.addWidget(self.test_button, 3, 0, 1, 2)

        self.loop_checkbox = QCheckBox("循环播放 A/B")
        playback_layout.addWidget(self.loop_checkbox, 4, 0)
        volume_row = QHBoxLayout()
        volume_row.addWidget(QLabel("播放音量"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(5, 80)
        self.volume_slider.setValue(70)
        volume_row.addWidget(self.volume_slider)
        playback_layout.addLayout(volume_row, 4, 1)

        reference_level_row = QHBoxLayout()
        reference_level_row.addWidget(QLabel("参考音电平"))
        self.reference_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.reference_level_slider.setRange(10, 100)
        self.reference_level_slider.setValue(70)
        self.reference_level_slider.setToolTip("只改变合成正弦的电平，不改变 GPU 录音")
        reference_level_row.addWidget(self.reference_level_slider)
        playback_layout.addLayout(reference_level_row, 5, 0, 1, 2)

        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("录音起点"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 600)
        self.start_spin.setDecimals(1)
        self.start_spin.setSuffix(" s")
        self.start_spin.setValue(start_seconds)
        settings_row.addWidget(self.start_spin)
        settings_row.addWidget(QLabel("片段长度"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 20)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(sample_seconds)
        settings_row.addWidget(self.duration_spin)
        settings_row.addStretch(1)
        playback_layout.addLayout(settings_row, 6, 0, 1, 2)
        outer.addWidget(playback)

        decision = QGroupBox("参考正弦相对 GPU 录音")
        decision_layout = QGridLayout(decision)
        high_button = QPushButton("参考音高了  (H)")
        low_button = QPushButton("参考音低了  (L)")
        match_button = QPushButton("听起来匹配 / 保存  (M)")
        undo_button = QPushButton("撤销一步  (U)")
        reset_button = QPushButton("重置为 20000 Hz")
        skip_button = QPushButton("跳过此录音")
        high_button.setObjectName("high")
        low_button.setObjectName("low")
        match_button.setObjectName("match")
        high_button.clicked.connect(self.reference_is_high)
        low_button.clicked.connect(self.reference_is_low)
        match_button.clicked.connect(self.accept_match)
        undo_button.clicked.connect(self.undo)
        reset_button.clicked.connect(self.reset_current)
        skip_button.clicked.connect(self.next_target)
        decision_layout.addWidget(high_button, 0, 0)
        decision_layout.addWidget(low_button, 0, 1)
        decision_layout.addWidget(match_button, 1, 0, 1, 2)
        decision_layout.addWidget(undo_button, 2, 0)
        decision_layout.addWidget(reset_button, 2, 1)
        decision_layout.addWidget(skip_button, 3, 0, 1, 2)
        outer.addWidget(decision)

        self.result_view = QPlainTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMaximumBlockCount(500)
        outer.addWidget(QLabel("已保存标注（按录音列表显示）"))
        outer.addWidget(self.result_view, 1)

        self.statusBar().showMessage("就绪")
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111417; color: #e8ecef; font-size: 14px; }
            QLabel#title { font-size: 28px; font-weight: 800; color: #f4f7f8; }
            QLabel#subtitle { color: #9ca9b0; margin-bottom: 6px; }
            QLabel#warning { color: #ff9f55; font-weight: 700; margin-bottom: 6px; }
            QLabel#frequency { font-size: 34px; font-weight: 800; color: #ffd34e; }
            QGroupBox { border: 1px solid #3b454b; margin-top: 9px; padding-top: 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #252c31; border: 1px solid #56636a; padding: 10px; min-height: 24px; }
            QPushButton:hover { background: #333d43; }
            QPushButton#high { background: #6b302d; border-color: #bd5d55; }
            QPushButton#low { background: #234f69; border-color: #4b91ba; }
            QPushButton#match { background: #315d37; border-color: #63a96c; font-weight: 800; }
            QComboBox, QDoubleSpinBox, QPlainTextEdit { background: #1b2024; border: 1px solid #465159; padding: 6px; }
            """
        )

    def install_shortcuts(self) -> None:
        bindings = {
            "R": self.play_reference,
            "G": self.play_recording,
            "Space": self.play_ab,
            "Escape": self.stop_audio,
            "H": self.reference_is_high,
            "L": self.reference_is_low,
            "M": self.accept_match,
            "U": self.undo,
        }
        for key, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    @property
    def current_file(self) -> Path:
        return self.files[self.current_index]

    def load_results(self) -> dict:
        if not self.output_path.exists():
            return {"version": 1, "results": {}}
        try:
            return json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "results": {}}

    def save_results(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.refresh_results()

    def refresh_results(self) -> None:
        lines = []
        for path in self.files:
            item = self.results.get("results", {}).get(str(path))
            if item:
                lines.append(f"{path.name:28s}  {item['frequency_hz']:9.2f} Hz")
            else:
                lines.append(f"{path.name:28s}  —")
        self.result_view.setPlainText("\n".join(lines))

    def add_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "添加 WAV", str(Path.cwd()), "WAV (*.wav)")
        if not names:
            return
        known = set(self.files)
        self.files.extend(Path(name).resolve() for name in names if Path(name).resolve() not in known)
        self.files.sort(key=numeric_key)
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for path in self.files:
            self.file_combo.addItem(path.name, path)
        self.file_combo.blockSignals(False)
        self.load_target(0)

    def load_target(self, index: int) -> None:
        if not self.files:
            return
        self.stop_audio()
        self.current_index = max(0, min(index, len(self.files) - 1))
        if self.file_combo.currentIndex() != self.current_index:
            self.file_combo.blockSignals(True)
            self.file_combo.setCurrentIndex(self.current_index)
            self.file_combo.blockSignals(False)
        self.reset_state()
        existing = self.results.get("results", {}).get(str(self.current_file))
        if existing:
            self.statusBar().showMessage(
                f"已有标注 {existing['frequency_hz']:.2f} Hz；可重新二分并覆盖"
            )
        else:
            self.statusBar().showMessage(f"当前：{self.current_file.name}")
        self.refresh_results()

    def reset_state(self) -> None:
        self.low_hz = MIN_FREQUENCY
        self.high_hz = MAX_FREQUENCY
        self.reference_hz = INITIAL_FREQUENCY
        self.history.clear()
        self.update_readout()

    def update_readout(self) -> None:
        self.frequency_label.setText(f"{self.reference_hz:,.2f} Hz")
        self.bounds_label.setText(f"{self.low_hz:,.2f} — {self.high_hz:,.2f} Hz（对数二分）")
        cents = 1200.0 * math.log2(self.high_hz / self.low_hz)
        self.iteration_label.setText(f"{len(self.history)} 次；剩余区间 {cents:.1f} 音分")

    def reset_current(self) -> None:
        self.stop_audio()
        self.reset_state()
        self.statusBar().showMessage("已重置为 20000 Hz")

    def generate_reference(self, frequency_hz: float | None = None) -> tuple[int, np.ndarray]:
        sample_rate = 48_000
        count = int(REFERENCE_SECONDS * sample_rate)
        time_axis = np.arange(count, dtype=np.float64) / sample_rate
        # A pure sine has far less broadband energy than the phone recording.
        # Give it an independent, clearly audible level instead of matching RMS.
        target_peak = 0.35 * self.reference_level_slider.value() / 100.0
        frequency = self.reference_hz if frequency_hz is None else frequency_hz
        tone = target_peak * np.sin(2.0 * np.pi * frequency * time_axis)
        fade_count = int(0.02 * sample_rate)
        fade = np.linspace(0.0, 1.0, fade_count)
        tone[:fade_count] *= fade
        tone[-fade_count:] *= fade[::-1]
        stereo = np.column_stack((tone, tone))
        return sample_rate, stereo

    def read_recording_segment(self) -> tuple[int, np.ndarray]:
        sample_rate, source = wavfile.read(self.current_file)
        source_dtype = source.dtype
        audio = source.astype(np.float64)
        if np.issubdtype(source_dtype, np.integer):
            audio /= max(abs(np.iinfo(source_dtype).min), np.iinfo(source_dtype).max)
        if audio.ndim == 1:
            audio = audio[:, None]
        start = int(self.start_spin.value() * sample_rate)
        end = int((self.start_spin.value() + self.duration_spin.value()) * sample_rate)
        segment = audio[start:end]
        if not segment.size:
            raise ValueError("所选录音片段为空，请调整起点或长度")
        return sample_rate, segment

    def selected_audio_device(self):
        return self.device_combo.currentData()

    def audio_device_changed(self) -> None:
        self.stop_audio()
        device = self.selected_audio_device()
        if device is not None:
            self.statusBar().showMessage(f"音频输出：{device.description()}")

    def stop_sink(self) -> None:
        if self.audio_sink is not None:
            sink = self.audio_sink
            self.audio_sink = None
            sink.stop()
            sink.deleteLater()
        if self.audio_buffer is not None:
            self.audio_buffer.close()
            self.audio_buffer.deleteLater()
            self.audio_buffer = None
        self.audio_bytes = None

    def stop_audio(self) -> None:
        self.sequence.clear()
        self.sequence_active = False
        self.stop_sink()
        self.statusBar().showMessage("已停止")

    def play_pcm(self, sample_rate: int, audio: np.ndarray) -> None:
        self.stop_sink()
        output_rate = 48_000
        if sample_rate != output_rate:
            divisor = math.gcd(sample_rate, output_rate)
            audio = resample_poly(audio, output_rate // divisor, sample_rate // divisor, axis=0)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = np.ascontiguousarray((pcm * 32767.0).astype("<i2"))
        audio_format = QAudioFormat()
        audio_format.setSampleRate(output_rate)
        audio_format.setChannelCount(2)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        device = self.selected_audio_device()
        if device is None or device.isNull():
            raise RuntimeError("没有可用的 Qt 音频输出设备")
        if not device.isFormatSupported(audio_format):
            raise RuntimeError(f"{device.description()} 不支持 48 kHz / 双声道 / PCM16")

        self.audio_bytes = QByteArray(pcm.tobytes())
        self.audio_buffer = QBuffer(self)
        self.audio_buffer.setData(self.audio_bytes)
        self.audio_buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        sink = QAudioSink(device, audio_format, self)
        sink.setVolume(self.volume_slider.value() / 100.0)
        sink.stateChanged.connect(lambda state, owner=sink: self.on_audio_state(owner, state))
        self.audio_sink = sink
        sink.start(self.audio_buffer)

    def report_playback_error(self, message: str) -> None:
        self.sequence_active = False
        self.statusBar().showMessage(f"播放失败：{message}")
        QMessageBox.warning(self, "Qt 音频播放失败", message)

    def play_reference(self) -> None:
        self.stop_audio()
        try:
            sample_rate, audio = self.generate_reference()
            self.play_pcm(sample_rate, audio)
            self.statusBar().showMessage(f"播放参考音：{self.reference_hz:.2f} Hz")
        except Exception as exc:
            self.report_playback_error(str(exc))

    def play_recording(self) -> None:
        self.stop_audio()
        try:
            sample_rate, audio = self.read_recording_segment()
            self.play_pcm(sample_rate, audio)
            self.statusBar().showMessage(f"播放 GPU 录音：{self.current_file.name}")
        except Exception as exc:
            self.report_playback_error(str(exc))

    def play_device_test(self) -> None:
        self.stop_audio()
        try:
            sample_rate, audio = self.generate_reference(1000.0)
            self.play_pcm(sample_rate, audio)
            self.statusBar().showMessage("播放 1000 Hz 设备测试；若无声请切换音频输出")
        except Exception as exc:
            self.report_playback_error(str(exc))

    def play_ab(self) -> None:
        self.stop_audio()
        self.sequence_active = True
        self.sequence = ["reference", "recording"]
        self.play_next_sequence_item()

    def play_next_sequence_item(self) -> None:
        if not self.sequence_active:
            return
        if not self.sequence:
            if self.loop_checkbox.isChecked():
                self.sequence = ["reference", "recording"]
            else:
                self.sequence_active = False
                self.statusBar().showMessage("A/B 播放完成")
                return
        item = self.sequence.pop(0)
        try:
            if item == "reference":
                sample_rate, audio = self.generate_reference()
                self.statusBar().showMessage(f"A / 参考音 {self.reference_hz:.2f} Hz")
            else:
                sample_rate, audio = self.read_recording_segment()
                self.statusBar().showMessage(f"B / GPU 录音 {self.current_file.name}")
            self.play_pcm(sample_rate, audio)
        except Exception as exc:
            self.report_playback_error(str(exc))

    def on_audio_state(self, owner: QAudioSink, state: QAudio.State) -> None:
        if owner is not self.audio_sink:
            return
        if state == QAudio.State.IdleState:
            self.stop_sink()
            if self.sequence_active:
                QTimer.singleShot(350, self.play_next_sequence_item)
        elif state == QAudio.State.StoppedState and owner.error() != QAudio.Error.NoError:
            error_name = owner.error().name
            self.stop_sink()
            self.report_playback_error(f"{self.selected_audio_device().description()}: {error_name}")

    def bisect(self) -> None:
        self.reference_hz = math.sqrt(self.low_hz * self.high_hz)
        self.update_readout()
        if self.loop_checkbox.isChecked():
            self.play_ab()

    def reference_is_high(self) -> None:
        self.stop_audio()
        self.history.append((self.low_hz, self.high_hz, self.reference_hz))
        self.high_hz = min(self.high_hz, self.reference_hz)
        self.bisect()
        self.statusBar().showMessage("已标记：参考音高了 → 下调参考频率")

    def reference_is_low(self) -> None:
        self.stop_audio()
        if self.reference_hz >= MAX_FREQUENCY - 0.5:
            QMessageBox.information(self, "到达上限", "参考音已接近 22 kHz / 48 kHz 采样上限。")
            return
        self.history.append((self.low_hz, self.high_hz, self.reference_hz))
        self.low_hz = max(self.low_hz, self.reference_hz)
        self.bisect()
        self.statusBar().showMessage("已标记：参考音低了 → 上调参考频率")

    def undo(self) -> None:
        self.stop_audio()
        if not self.history:
            self.statusBar().showMessage("没有可撤销的判断")
            return
        self.low_hz, self.high_hz, self.reference_hz = self.history.pop()
        self.update_readout()
        self.statusBar().showMessage("已撤销一步")

    def accept_match(self) -> None:
        self.stop_audio()
        self.results.setdefault("results", {})[str(self.current_file)] = {
            "frequency_hz": round(self.reference_hz, 4),
            "bounds_hz": [round(self.low_hz, 4), round(self.high_hz, 4)],
            "decisions": len(self.history),
            "recording_start_seconds": self.start_spin.value(),
            "recording_duration_seconds": self.duration_spin.value(),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self.save_results()
        self.statusBar().showMessage(f"已保存：{self.current_file.name} = {self.reference_hz:.2f} Hz")
        QTimer.singleShot(300, self.next_target)

    def next_target(self) -> None:
        if not self.files:
            return
        next_index = (self.current_index + 1) % len(self.files)
        self.file_combo.setCurrentIndex(next_index)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_audio()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt6 音高二分标注器")
    parser.add_argument("files", nargs="*", type=Path, help="默认加载 denoised/*_denoised.wav")
    parser.add_argument("--output", type=Path, default=Path("pitch_labels.json"))
    parser.add_argument("--start", type=float, default=2.0, help="录音试听起点秒数")
    parser.add_argument("--seconds", type=float, default=3.0, help="录音试听长度")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args.files)
    if not files:
        raise SystemExit("没有找到 WAV；可显式传入文件，或先生成 denoised/*.wav")
    app = QApplication(sys.argv)
    window = PitchAnnotator(files, args.output, args.start, args.seconds)
    window.show()
    if args.smoke_test:
        QTimer.singleShot(250, app.quit)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
