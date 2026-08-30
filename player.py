#!/usr/bin/env python3
"""Play note sequences by mapping pitch to dual-V100 FP32 GEMM sizes."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class Anchor:
    size: int
    frequency_hz: float


@dataclass(frozen=True)
class Event:
    label: str
    frequency_hz: float | None
    beats: float
    seconds: float | None = None


def merge_rests(events: Sequence[Event]) -> list[Event]:
    """Turn inter-note rests into sustain time for a legato audition."""
    merged: list[Event] = []
    for index, event in enumerate(events):
        is_rest = event.frequency_hz is None
        has_next_note = any(next_event.frequency_hz is not None for next_event in events[index + 1 :])
        if is_rest and merged and merged[-1].frequency_hz is not None and has_next_note:
            previous = merged[-1]
            if previous.seconds is not None and event.seconds is not None:
                merged[-1] = Event(previous.label, previous.frequency_hz, previous.beats,
                                    previous.seconds + event.seconds)
            else:
                merged[-1] = Event(previous.label, previous.frequency_hz,
                                    previous.beats + event.beats, previous.seconds)
        else:
            merged.append(event)
    return merged


def note_frequency(note: str, transpose: int = 0) -> float:
    match = NOTE_PATTERN.match(note.strip())
    if not match:
        raise ValueError(f"无效音名: {note}；示例 C5、F#5、Bb4")
    letter, accidental, octave_text = match.groups()
    semitone = NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = (int(octave_text) + 1) * 12 + semitone + transpose
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def load_anchors(path: Path) -> list[Anchor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = []
    for filename, item in payload.get("results", {}).items():
        match = re.search(r"\d+", Path(filename).stem)
        if match:
            anchors.append(Anchor(int(match.group()), float(item["frequency_hz"])))
    anchors.sort(key=lambda item: item.frequency_hz)
    if len(anchors) < 2:
        raise SystemExit("pitch_labels.json 至少需要两个有效标注")
    frequencies = [item.frequency_hz for item in anchors]
    sizes = [item.size for item in anchors]
    if any(a >= b for a, b in zip(frequencies, frequencies[1:])):
        raise SystemExit("标注频率必须能严格排序")
    if any(a <= b for a, b in zip(sizes, sizes[1:])):
        raise SystemExit("当前模型要求频率升高时矩阵尺寸严格减小")
    return anchors


def size_for_frequency(frequency_hz: float, anchors: Sequence[Anchor]) -> int:
    minimum = anchors[0].frequency_hz
    maximum = anchors[-1].frequency_hz
    if not minimum <= frequency_hz <= maximum:
        raise ValueError(
            f"{frequency_hz:.2f} Hz 超出已标注音域 {minimum:.2f}–{maximum:.2f} Hz"
        )
    log_frequencies = np.log([item.frequency_hz for item in anchors])
    log_sizes = np.log([item.size for item in anchors])
    interpolated = float(np.interp(math.log(frequency_hz), log_frequencies, log_sizes))
    size = int(round(math.exp(interpolated) / 8.0) * 8)
    return max(8, size)


def load_tuning(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"notes": {}, "events": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tuning JSON 顶层必须是对象")
    payload.setdefault("notes", {})
    payload.setdefault("events", {})
    return payload


def tuned_size(
    event: Event,
    event_index: int,
    anchors: Sequence[Anchor],
    tuning: dict,
) -> tuple[int, str]:
    assert event.frequency_hz is not None
    base_size = size_for_frequency(event.frequency_hz, anchors)
    rules = {}
    note_rule = tuning.get("notes", {}).get(event.label, {})
    event_rule = tuning.get("events", {}).get(str(event_index), {})
    if isinstance(note_rule, dict):
        rules.update(note_rule)
    if isinstance(event_rule, dict):
        rules.update(event_rule)

    details = []
    if "size" in rules:
        size = int(rules["size"])
        details.append(f"size={size}")
    elif "semitones" in rules:
        semitones = float(rules["semitones"])
        compensated_frequency = event.frequency_hz * 2.0 ** (semitones / 12.0)
        size = size_for_frequency(compensated_frequency, anchors)
        details.append(f"{semitones:+g}st")
    else:
        size = base_size

    if "size_offset" in rules:
        offset = int(rules["size_offset"])
        size += offset
        details.append(f"size{offset:+d}")

    size = int(round(size / 8.0) * 8)
    minimum_size = min(anchor.size for anchor in anchors)
    maximum_size = max(anchor.size for anchor in anchors)
    if not minimum_size <= size <= maximum_size:
        raise ValueError(
            f"事件 {event_index} / {event.label} 微调后尺寸 {size} 超出 "
            f"{minimum_size}–{maximum_size}"
        )
    suffix = f" [tune {' '.join(details)}]" if details else ""
    return size, suffix


def chromatic_scale(transpose: int) -> list[Event]:
    names = ["C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5", "C6"]
    return [Event(name, note_frequency(name, transpose), 1.0) for name in names]


def anchor_scale(anchors: Sequence[Anchor]) -> list[Event]:
    return [
        Event(f"anchor-{anchor.size}", anchor.frequency_hz, 1.0)
        for anchor in anchors
    ]


def load_score(path: Path, transpose: int) -> list[Event]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload["events"] if isinstance(payload, dict) else payload
    events = []
    for index, item in enumerate(raw_events):
        seconds = float(item["seconds"]) if "seconds" in item else None
        beats = float(item.get("beats", 1.0))
        if beats <= 0 or (seconds is not None and seconds <= 0):
            raise ValueError(f"第 {index + 1} 个事件 beats/seconds 必须大于 0")
        note = str(item.get("note", "rest"))
        if note.lower() in ("rest", "r", "-"):
            events.append(Event("rest", None, beats, seconds))
        elif "frequency_hz" in item:
            events.append(Event(note, float(item["frequency_hz"]), beats, seconds))
        else:
            events.append(Event(note, note_frequency(note, transpose), beats, seconds))
    return events


def parse_devices(value: str, count: int) -> list[torch.device]:
    try:
        indices = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--devices 格式应为 0,1") from exc
    if len(indices) != 2 or len(set(indices)) != 2:
        raise SystemExit("播放器需要两张不同 GPU，例如 --devices 0,1")
    if any(index < 0 or index >= count for index in indices):
        raise SystemExit(f"CUDA 设备越界；当前共有 {count} 张")
    return [torch.device(f"cuda:{index}") for index in indices]


class DualV100Player:
    def __init__(self, devices: Sequence[torch.device], repeats: int, seed: int) -> None:
        self.devices = devices
        self.repeats = repeats
        self.seed = seed
        self.cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def prepare(self, sizes_by_device: Sequence[Sequence[int]]) -> None:
        for device_index, sizes in enumerate(sizes_by_device):
            device = self.devices[device_index]
            for size in sorted(set(sizes)):
                key = (size, device_index)
                if key in self.cache:
                    continue
                working_set = 3 * size * size * 4 / 2**20
                print(f"准备 GPU{device_index + 1} {size}x{size}：{working_set:.1f} MiB")
                generator = torch.Generator(device="cpu").manual_seed(self.seed + size)
                host_a = torch.randn((size, size), dtype=torch.float32, generator=generator)
                host_b = torch.randn((size, size), dtype=torch.float32, generator=generator)
                a = host_a.to(device)
                b = host_b.to(device)
                output = torch.empty_like(a)
                self.cache[key] = (a, b, output)
                with torch.inference_mode():
                    torch.mm(a, b, out=output)
                torch.cuda.synchronize(device)

    def play_note(self, size: int, seconds: float) -> None:
        matrices = [self.cache[(size, device_index)] for device_index in range(len(self.devices))]
        started = time.monotonic()
        with torch.inference_mode():
            while time.monotonic() - started < seconds:
                for _ in range(self.repeats):
                    for a, b, output in matrices:
                        torch.mm(a, b, out=output)
                for device in self.devices:
                    torch.cuda.synchronize(device)

    def play_chord(self, sizes: tuple[int, int], seconds: float) -> None:
        """Run one independently mapped matrix size on each GPU."""
        matrices = [self.cache[(sizes[0], 0)], self.cache[(sizes[1], 1)]]
        started = time.monotonic()
        with torch.inference_mode():
            while time.monotonic() - started < seconds:
                for _ in range(self.repeats):
                    for a, b, output in matrices:
                        torch.mm(a, b, out=output)
                for device in self.devices:
                    torch.cuda.synchronize(device)

    def play(
        self,
        events: Sequence[Event],
        anchors: Sequence[Anchor],
        tuning: dict,
        bpm: float,
    gap: float,
    chord_semitones: int | None = None,
    legato: bool = False,
    ) -> None:
        if legato:
            events = merge_rests(events)
        beat_seconds = 60.0 / bpm
        mapped = []
        for index, event in enumerate(events, 1):
            if event.frequency_hz is None:
                mapped.append((event, None, "", None))
            else:
                size, suffix = tuned_size(event, index, anchors, tuning)
                companion = None
                if chord_semitones is not None:
                    companion_frequency = event.frequency_hz * 2.0 ** (chord_semitones / 12.0)
                    companion = size_for_frequency(companion_frequency, anchors)
                mapped.append((event, size, suffix, companion))
        if chord_semitones is None:
            primary = [size for _, size, _, _ in mapped if size is not None]
            self.prepare([primary, primary])
        else:
            primary = [size for _, size, _, _ in mapped if size is not None]
            companion_sizes = [companion for _, _, _, companion in mapped if companion is not None]
            self.prepare([primary, companion_sizes])

        print("\n开始播放，Ctrl-C 停止")
        for index, (event, size, suffix, companion) in enumerate(mapped, 1):
            duration = event.seconds if event.seconds is not None else event.beats * beat_seconds
            sounding = max(0.0, duration - gap)
            if size is None:
                print(f"[{index:02d}/{len(mapped):02d}] rest {duration:.3f}s")
                time.sleep(duration)
                continue
            assert event.frequency_hz is not None
            print(
                f"[{index:02d}/{len(mapped):02d}] {event.label:10s} "
                f"{event.frequency_hz:8.2f} Hz -> GPU1 {size:4d}x{size}{suffix}"
            )
            if companion is not None:
                print(f"    chord {chord_semitones:+d}st -> GPU2 {companion:4d}x{companion}")
            if sounding > 0:
                try:
                    if companion is not None:
                        self.play_chord((size, companion), sounding)
                    else:
                        self.play_note(size, sounding)
                except RuntimeError as exc:
                    raise RuntimeError(f"事件 {index} ({event.label}) GPU 计算失败：{exc}") from exc
            if gap > 0:
                time.sleep(min(gap, duration))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用双 V100 FP32 GEMM 播放音符序列")
    parser.add_argument("demo", nargs="?", choices=["scale", "anchors"], default="scale")
    parser.add_argument("--labels", type=Path, default=Path("pitch_labels.json"))
    parser.add_argument("--score", type=Path, help="JSON 乐谱；指定后忽略 demo")
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("tuning.json"),
        help="微调 JSON，默认自动读取 tuning.json",
    )
    parser.add_argument("--no-tuning", action="store_true", help="忽略所有微调")
    parser.add_argument("--bpm", type=float, default=150.0)
    parser.add_argument("--gap", type=float, default=0.06, help="音符间断音秒数，默认 0.06")
    parser.add_argument("--chord-semitones", type=int, metavar="N",
                        help="双卡和弦：GPU2 相对 GPU1 移调 N 个半音，例如 -7 为低五度")
    parser.add_argument("--legato", action="store_true",
                        help="把音符之间的休止并入前一个音符，连续运行不间断")
    parser.add_argument("--transpose", type=int, default=0, help="乐谱移调半音数")
    parser.add_argument("--repeats", type=int, default=4, help="每轮同步前每卡 GEMM 数")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="只打印映射，不运行 GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bpm <= 0 or args.gap < 0 or args.repeats <= 0:
        raise SystemExit("--bpm/--repeats 必须大于 0，--gap 不能小于 0")
    anchors = load_anchors(args.labels)
    tuning = load_tuning(None if args.no_tuning else args.tuning)
    if args.score:
        events = load_score(args.score, args.transpose)
    elif args.demo == "anchors":
        events = anchor_scale(anchors)
    else:
        events = chromatic_scale(args.transpose)
    if args.legato:
        events = merge_rests(events)

    print("标注锚点（低 -> 高）：")
    for anchor in anchors:
        print(f"  {anchor.frequency_hz:8.2f} Hz -> {anchor.size}x{anchor.size}")
    print("\n事件映射：")
    for index, event in enumerate(events, 1):
        if event.frequency_hz is None:
            print(f"  {event.label:10s} -> rest")
        else:
            size, suffix = tuned_size(event, index, anchors, tuning)
            if args.chord_semitones is None:
                print(f"  {event.label:10s} {event.frequency_hz:8.2f} Hz -> {size}x{size}{suffix}")
            else:
                companion_frequency = event.frequency_hz * 2.0 ** (args.chord_semitones / 12.0)
                companion = size_for_frequency(companion_frequency, anchors)
                print(f"  {event.label:10s} {event.frequency_hz:8.2f} Hz -> GPU1 {size}x{size}; GPU2 {companion}x{companion} ({args.chord_semitones:+d}st)")
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise SystemExit("没有可用 CUDA GPU；可先用 --dry-run 检查映射")

    devices = parse_devices(args.devices, torch.cuda.device_count())
    player = DualV100Player(devices, args.repeats, args.seed)
    try:
        player.play(events, anchors, tuning, args.bpm, args.gap, args.chord_semitones, args.legato)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
