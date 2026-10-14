#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Drive audible-rate GPU activity modulation; acoustic pitch needs measurement."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from player import load_score, merge_rests


CARRIERS = {"compute": 0, "memory": 1, "mixed": 2, "tensor": 3, "fp4": 4}


@dataclass(frozen=True)
class Segment:
    label: str
    seconds: float
    frequencies: tuple[float | None, ...]


def load_dual_score(path: Path, transpose: int) -> list[Segment]:
    """Load a score whose events carry independent GPU voice frequencies."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("mode") == "gpu_activity_modulation":
        raise ValueError(f"{path} 是播放诊断日志，不是双卡乐谱；请从 MIDI 重新生成乐谱")
    raw_events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list):
        raise ValueError("双卡谱必须包含 events 数组")
    segments = []
    for index, item in enumerate(raw_events):
        try:
            seconds = float(item["seconds"])
            values = item["frequencies_hz"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"双卡谱第 {index + 1} 项缺少 seconds/frequencies_hz") from exc
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"双卡谱第 {index + 1} 项必须有两个频率")
        frequencies = tuple(
            None if value is None else float(value) * 2 ** (transpose / 12)
            for value in values
        )
        label = str(item.get("label", "dual"))
        segments.append(Segment(label, seconds, frequencies))
    return segments


def make_schedule(args: argparse.Namespace, card_count: int) -> list[Segment]:
    if args.chord_semitones is not None and card_count != 2:
        raise ValueError("--chord-semitones 需要两张卡")

    def voices(frequency: float | None) -> tuple[float | None, ...]:
        if frequency is None:
            return (None,) * card_count
        result = [frequency] * card_count
        if args.chord_semitones is not None:
            result[1] *= 2 ** (args.chord_semitones / 12)
        return tuple(result)

    segments = []
    if getattr(args, "dual_score", None):
        if card_count != 2:
            raise ValueError("--dual-score 需要两张卡")
        if args.chord_semitones is not None or args.legato:
            raise ValueError("双卡谱保留各自休止；请省略 --chord-semitones 和 --legato")
        segments = load_dual_score(args.dual_score, args.transpose)
    elif args.score:
        # Explicit frequency_hz is also transposed, unlike the legacy player.
        events = load_score(args.score, 0)
        if args.legato:
            events = merge_rests(events)
        for event in events:
            seconds = event.seconds if event.seconds is not None else event.beats * 60 / args.bpm
            frequency = event.frequency_hz
            if frequency is not None:
                frequency *= 2 ** (args.transpose / 12)
            label = event.label
            if frequency is not None and args.transpose:
                label += f" ({args.transpose:+d}st)"
            segments.append(Segment(label, seconds, voices(frequency)))
    else:
        frequencies = [args.frequency]
        if args.sweep:
            frequencies = [float(item.strip()) for item in args.sweep.split(",")]
        for index, frequency in enumerate(frequencies):
            if index and args.rest:
                segments.append(Segment("rest", args.rest, voices(None)))
            segments.append(Segment(f"{frequency:g} Hz", args.duration, voices(frequency)))
    for segment in segments:
        if not math.isfinite(segment.seconds) or segment.seconds <= 0:
            raise ValueError("每段时长必须为有限正数")
        for frequency in segment.frequencies:
            if frequency is not None and not (40 <= frequency <= 10000):
                raise ValueError(f"调制频率 {frequency:g} Hz 超出本实验的 40–10000 Hz 范围")
    if not segments:
        raise ValueError("乐谱没有可播放的事件")
    return segments


def voice_schedule(segments: list[Segment], card: int) -> list[Segment]:
    """A change in the other voice must not restart this card's pitch clock."""
    result = []
    for segment in segments:
        if result and result[-1].frequencies[card] == segment.frequencies[card]:
            previous = result[-1]
            result[-1] = Segment(previous.label, previous.seconds + segment.seconds, previous.frequencies)
        else:
            result.append(segment)
    return result


def timing_arrays(segments: list[Segment], card: int, duty: float) -> tuple[np.ndarray, ...]:
    periods, highs, ends = [], [], []
    elapsed = 0.0
    for segment in segments:
        frequency = segment.frequencies[card]
        period = round(1e9 / frequency) if frequency is not None else 0
        periods.append(period)
        highs.append(round(period * duty))
        elapsed += segment.seconds
        ends.append(round(elapsed * 1e9))
    return tuple(np.array(values, dtype=np.int64) for values in (periods, highs, ends))


def load_engine(devices: list[int]):
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").exists():
        raise RuntimeError("本模式需要 CUDA Toolkit 的 nvcc，只有 CUDA 版 torch 不够")
    root = Path(__file__).resolve().parent
    build_dir = root / ".torch_extensions" / "power_tone"
    build_dir.mkdir(parents=True, exist_ok=True)
    architectures = {"%d.%d" % torch.cuda.get_device_capability(device) for device in devices}
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(sorted(architectures)))
    os.environ.setdefault("MAX_JOBS", "2")
    print("加载 GPU 计时内核（首次运行需要编译）……", flush=True)
    return load(
        name="v100_power_tone", sources=[str(root / "native/power_tone.cu")],
        build_directory=str(build_dir), extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O2"], verbose=False,
    )


def play_start_marker() -> float:
    from probe import make_marker

    path = Path(make_marker())
    try:
        for command in (("paplay", str(path)), ("aplay", "-q", str(path))):
            if shutil.which(command[0]):
                started = time.time()
                try:
                    result = subprocess.run(command, capture_output=True, timeout=8)
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if result.returncode == 0:
                    time.sleep(0.5)
                    return started
        raise RuntimeError("开场定位音播放失败，请检查系统默认音频输出")
    finally:
        path.unlink(missing_ok=True)


class LinkLoad:
    """Paced bidirectional UVA copies with immutable sources, not an allreduce."""

    def __init__(self, engine, devices: list[int], gib_s: float, chunk_mib: int):
        self.engine = engine
        self.devices = devices
        self.chunk_bytes = chunk_mib * 2**20
        self.interval = 2 * self.chunk_bytes / (gib_s * 2**30)
        self.sent_bytes = 0
        self.pairs = 0
        self.next_copy = 0.0
        self.buffers = []
        for rank, device in enumerate(devices):
            with torch.cuda.device(device):
                source = torch.arange(256, dtype=torch.uint8, device=device).repeat(self.chunk_bytes // 256)
                source ^= rank * 0x55
                destination = torch.empty_like(source)
                stream = torch.cuda.Stream(device=device)
                torch.cuda.synchronize(device)
            self.buffers.append((stream, source, destination))
        # Verify both directions using the same transport used during playback.
        self._copy_pair()
        self.synchronize()
        for rank, device in enumerate(devices):
            with torch.cuda.device(device):
                expected = torch.arange(256, dtype=torch.uint8, device=device).repeat(self.chunk_bytes // 256)
                expected ^= (1 - rank) * 0x55
                if not torch.equal(self.buffers[rank][2], expected):
                    raise RuntimeError(f"cuda:{device} 双卡拷贝校验失败")
        self.sent_bytes = self.pairs = 0

    def _copy_pair(self) -> None:
        for rank, device in enumerate(self.devices):
            stream, _, destination = self.buffers[rank]
            with torch.cuda.device(device), torch.cuda.stream(stream):
                self.engine.copy_link(destination, self.buffers[1 - rank][1])

    def poll(self, now: float) -> None:
        if now < self.next_copy or not all(item[0].query() for item in self.buffers):
            return
        self._copy_pair()
        self.pairs += 1
        self.sent_bytes += self.chunk_bytes * 2
        self.next_copy = max(self.next_copy + self.interval, now - self.interval)

    def synchronize(self) -> None:
        for stream, *_ in self.buffers:
            stream.synchronize()


class BurstPlayer:
    """Replay GPU-timed finite bursts, keeping three graph batches queued."""

    def __init__(self, engine, buffers, stop, args):
        self.engine = engine
        self.device, self.stream, arrays, memory, sink, self.stats, edges, blocks = buffers
        self.stop = stop
        self.state = torch.zeros(6, dtype=torch.int64, device=self.device)
        self.done_view = self.state[:1]
        self.graph = torch.cuda.CUDAGraph()
        # Schedule progression and all 32 per-cycle waits run on the GPU.
        with torch.cuda.graph(self.graph, stream=self.stream):
            for _ in range(32):
                engine.burst_cycle(*arrays, memory, sink, self.stats, edges, self.state,
                                   stop, blocks, CARRIERS[args.carrier],
                                   args.work_repeats, args.tensor_reuse)
        self.slots = [(torch.empty(1, dtype=torch.int64, pin_memory=True), torch.cuda.Event())
                      for _ in range(3)]
        self.pending = deque()
        self.finished = False

    def _enqueue(self, slot):
        done, event = self.slots[slot]
        self.graph.replay()
        done.copy_(self.done_view, non_blocking=True)
        event.record(self.stream)
        self.pending.append(slot)

    def start(self):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.engine.start_burst(self.stats, self.state)
            for slot in range(len(self.slots)):
                self._enqueue(slot)

    def poll(self):
        if self.finished:
            return
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            while self.pending and self.slots[self.pending[0]][1].query():
                slot = self.pending.popleft()
                if self.slots[slot][0].item():
                    self.finished = True
                    return
                self._enqueue(slot)


def run(args: argparse.Namespace, segments: list[Segment], devices: list[int]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU；可先用 --dry-run 查看调制时间表")
    if any(device < 0 or device >= torch.cuda.device_count() for device in devices):
        raise ValueError("--devices 包含不存在的 GPU")
    engine_mode = getattr(args, "engine", "resident")
    if engine_mode == "burst" and args.carrier not in ("tensor", "fp4"):
        raise ValueError("--engine burst 需要 --carrier tensor 或 fp4")
    engine = load_engine(devices)
    stop = engine.StopFlag()
    buffers = []
    card_schedules = [voice_schedule(segments, card) for card in range(len(devices))]
    for card, device in enumerate(devices):
        props = torch.cuda.get_device_properties(device)
        if props.major < 7:
            raise ValueError("本内核的 __nanosleep 需要 compute capability >= 7.0")
        resident_limit = engine.resident_blocks_per_sm(device, CARRIERS[args.carrier], engine_mode == "burst")
        blocks_per_sm = min(args.blocks_per_sm, resident_limit)
        blocks = props.multi_processor_count * blocks_per_sm
        # Use a disjoint ring per thread (FP32) or per warp (packed FP4).
        tensor_carrier = args.carrier in ("tensor", "fp4")
        memory_stride = blocks * 8 * 160 if tensor_carrier else blocks * 256 * 16
        requested_bytes = args.memory_mib * 2**20
        memory_bytes = math.ceil(requested_bytes / memory_stride) * memory_stride
        with torch.cuda.device(device):
            available, _ = torch.cuda.mem_get_info()
            reserve_bytes = max(256 * 2**20, 4 * args.link_chunk_mib * 2**20 if args.link_gib_s else 0)
            if memory_bytes + reserve_bytes > available:
                raise ValueError(f"cuda:{device}: 工作集需要 {memory_bytes / 2**30:.2f} GiB，"
                                 f"当前空闲 {available / 2**30:.2f} GiB；请减小 --memory-mib")
            stream = torch.cuda.Stream(device=device)
            arrays = [torch.from_numpy(values).to(device)
                      for values in timing_arrays(card_schedules[card], card, args.duty)]
            if tensor_carrier:
                memory = torch.empty(memory_bytes, dtype=torch.uint8, device=device)
                engine.fill_weights(memory)
            else:
                memory = torch.ones(memory_bytes // 4, dtype=torch.float32, device=device)
            sink = torch.zeros(blocks * 256, dtype=torch.float32, device=device)
            stats = torch.zeros(6, dtype=torch.int64, device=device)
            edges = torch.full((4096, 2), -1, dtype=torch.int64, device=device)
            torch.cuda.synchronize(device)
        buffers.append((device, stream, arrays, memory, sink, stats, edges, blocks))
        print(f"cuda:{device}: {props.name} / {blocks} blocks / {memory_bytes / 2**20:g} MiB", flush=True)
        if blocks_per_sm != args.blocks_per_sm:
            print(f"  每个 SM 限为 {blocks_per_sm} blocks，确保整张卡同时调制", flush=True)
    link = LinkLoad(engine, devices, args.link_gib_s, args.link_chunk_mib) if args.link_gib_s else None
    if link:
        print(f"双卡拷贝校验通过；双向合计目标 {args.link_gib_s:g} GiB/s（持续背景负载）", flush=True)
    bursts = []
    if engine_mode == "burst":
        print("构建 CUDA Graph：忙半周期完成后退出计算内核，闲半周期只用一个线程计时", flush=True)
        for buffer in buffers:
            with torch.cuda.device(buffer[0]):
                bursts.append(BurstPlayer(engine, buffer, stop, args))
    marker_unix = play_start_marker() if args.lineout_marker else None
    if args.countdown:
        print(f"{args.countdown:g} 秒后开始，可以用手机录音……", flush=True)
        time.sleep(args.countdown)
    launched = time.time()
    print("开始：GPU 按指定 Hz 切换计算活动；Ctrl-C 停止", flush=True)
    launch_times = []
    started_monotonic = time.monotonic()
    if link:
        link.next_copy = started_monotonic
    try:
        for card, (device, stream, arrays, memory, sink, stats, edges, blocks) in enumerate(buffers):
            with torch.cuda.device(device), torch.cuda.stream(stream):
                launch_times.append(time.time())
                if bursts:
                    bursts[card].start()
                else:
                    engine.launch(*arrays, memory, sink, stats, edges, stop, blocks,
                                  CARRIERS[args.carrier], args.work_repeats, args.tensor_reuse)
        # Poll streams so Ctrl-C can update the mapped stop flag while kernels run.
        while not (all(burst.finished for burst in bursts) if bursts
                   else all(stream.query() for _, stream, *_ in buffers)):
            for burst in bursts:
                burst.poll()
            if link:
                link.poll(time.monotonic())
            time.sleep(min(0.001 if bursts else 0.005, link.interval / 2) if link
                       else 0.001 if bursts else 0.03)
    except KeyboardInterrupt:
        stop.stop()
        print("正在停止 GPU 调制……", flush=True)
    finally:
        stop.stop()
        for _, stream, *_ in buffers:
            stream.synchronize()
        if link:
            link.synchronize()
    run_seconds = time.monotonic() - started_monotonic
    link_metrics = {"target_aggregate_gib_s": args.link_gib_s,
                    "copied_bytes": link.sent_bytes if link else 0,
                    "observed_aggregate_gib_s": link.sent_bytes / 2**30 / run_seconds if link else 0}
    if link:
        print(f"双向拷贝合计 {link_metrics['observed_aggregate_gib_s']:.3f} GiB/s")
    diagnostics = []
    for device, stream, arrays, memory, sink, stats, edges, blocks in buffers:
        values = stats.cpu().tolist()
        edge_values = edges.cpu().numpy()
        card_metrics = []
        card = devices.index(device)
        for event, segment in enumerate(card_schedules[card]):
            times = edge_values[edge_values[:, 0] == event, 1]
            if len(times) < 3 or segment.frequencies[card] is None:
                continue
            intervals = np.diff(times)
            measured = 1e9 / float(np.mean(intervals))
            jitter_us = float(np.std(intervals)) / 1000
            card_metrics.append({"event": event, "target_hz": segment.frequencies[card],
                                 "observed_switch_hz": measured, "interval_std_us": jitter_us,
                                 "edges_observed": len(times)})
        diagnostics.append({"device": device, "elapsed_s": values[1] / 1e9,
                            "active_iterations_thread0": values[2], "rising_edges_thread0": values[3],
                            "cancelled": bool(values[4]), "weight_tiles_warp0": values[5],
                            "allocated_work_bytes": memory.numel() * memory.element_size(),
                            "switch_measurements": card_metrics})
        print(f"cuda:{device}: {values[1]/1e9:.3f}s, 活动轮数 {values[2]}, 检测上升沿 {values[3]}")
        if args.carrier == "fp4":
            print(f"  采样 warp 已解码 {values[5]} 个权重 tile")
        for metric in card_metrics[:5]:
            print(f"  目标 {metric['target_hz']:.2f} Hz -> GPU 活动切换 {metric['observed_switch_hz']:.2f} Hz"
                  f"，周期标准差 {metric['interval_std_us']:.2f} µs")
        for metric in card_metrics:
            if abs(metric["observed_switch_hz"] / metric["target_hz"] - 1) > 0.01:
                print(f"  事件 {metric['event']}: 采样切换率偏差超过 1%；"
                      "可降低目标频率、降低 --tensor-reuse 或改用 --carrier tensor")
    print("以上测量是 GPU 活动切换节奏；实际电流波形和声音频率需另行测量。")
    timeline = []
    offset = 0.0
    for segment in segments:
        timeline.append({"label": segment.label, "start_s": offset, "end_s": offset + segment.seconds,
                         "frequencies_hz": segment.frequencies})
        offset += segment.seconds
    voice_timelines = []
    for card, schedule in enumerate(card_schedules):
        offset = 0.0
        timeline_voice = []
        for event, segment in enumerate(schedule):
            timeline_voice.append({"event": event, "start_s": offset,
                                   "end_s": offset + segment.seconds,
                                   "frequency_hz": segment.frequencies[card]})
            offset += segment.seconds
        voice_timelines.append({"device": devices[card], "segments": timeline_voice})
    return {"mode": "gpu_activity_modulation", "engine": engine_mode,
            "carrier": args.carrier, "duty": args.duty,
            "blocks_per_sm": args.blocks_per_sm, "work_repeats": args.work_repeats,
            "tensor_reuse": args.tensor_reuse, "link_load": link_metrics,
            "marker_unix": marker_unix, "launch_unix": launched,
            "kernel_start_delay_s": 0.002,
            "card_launch_unix": launch_times, "segments": timeline,
            "voice_timelines": voice_timelines, "diagnostics": diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 GPU 内按目标 Hz 调制负载，实验性供电啸叫发声")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frequency", type=float, default=1000.0, help="单音调制 Hz，默认 1000")
    group.add_argument("--sweep", help="逐个试音，例如 440,660,880,1320")
    group.add_argument("--score", type=Path, help="JSON 乐谱，直接用音符频率调制")
    group.add_argument("--dual-score", type=Path,
                       help="双声部 JSON 乐谱；frequencies_hz[0]/[1] 分别给两张卡")
    parser.add_argument("--duration", type=float, default=3.0, help="单音/扫频每段秒数")
    parser.add_argument("--rest", type=float, default=1.0, help="扫频段间休止秒数")
    parser.add_argument("--carrier", choices=CARRIERS, default="mixed", help="高负载类型，默认混合计算与显存")
    parser.add_argument("--engine", choices=("resident", "burst"), default="resident",
                        help="resident: 驻留内核；burst: 每个忙半周期退出内核，仅支持 tensor/fp4")
    parser.add_argument("--duty", type=float, default=0.5, help="高负载时间比例，默认 0.5；1 为连续负载对照")
    parser.add_argument("--devices", default="0,1", help="一张或两张 GPU，例如 0 或 0,1")
    parser.add_argument("--blocks-per-sm", type=int, default=4)
    parser.add_argument("--work-repeats", type=int, default=128,
                        help="每轮运算上限；FP4 模式是连续解码 tile 的上限，默认 128")
    parser.add_argument("--tensor-reuse", type=int, default=8,
                        help="FP4 模式每个解码 tile 的 Tensor Core 运算重复上限，默认 8")
    parser.add_argument("--memory-mib", type=int, default=64, help="每张卡的工作集 MiB；FP4 模式会轮流读它")
    parser.add_argument("--link-gib-s", type=float, default=0,
                        help="双向合计 GPU 拷贝目标 GiB/s，默认 0（关闭），不是 allreduce")
    parser.add_argument("--link-chunk-mib", type=int, default=4)
    parser.add_argument("--chord-semitones", type=int, help="第二张卡相对旋律的半音差")
    parser.add_argument("--transpose", type=int, default=0, help="乐谱移调半音，含显式 frequency_hz")
    parser.add_argument("--legato", action="store_true", help="合并音符间休止，保持连续旋律")
    parser.add_argument("--bpm", type=float, default=150)
    parser.add_argument("--countdown", type=float, default=3)
    parser.add_argument("--lineout-marker", action="store_true", help="开场只播放一次定位音")
    parser.add_argument("--manifest", type=Path, help="保存调制时间表及 GPU 计时诊断")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.manifest:
            for source in (args.score, args.dual_score):
                if source and (args.manifest.resolve() == source.resolve()
                               or (args.manifest.exists() and source.exists()
                                   and args.manifest.samefile(source))):
                    raise ValueError("--manifest 不能覆盖输入乐谱；请用独立的日志文件名，例如 senbonzakura_full_take.json")
        devices = [int(value.strip()) for value in args.devices.split(",")]
        if not 1 <= len(devices) <= 2 or len(set(devices)) != len(devices):
            raise ValueError("请指定一张或两张不同 GPU")
        if not math.isfinite(args.duty) or not 0 < args.duty <= 1:
            raise ValueError("--duty 必须在 (0, 1] 内")
        if not math.isfinite(args.bpm) or args.bpm <= 0:
            raise ValueError("--bpm 必须为有限正数")
        if not math.isfinite(args.countdown) or args.countdown < 0 or not math.isfinite(args.rest) or args.rest < 0:
            raise ValueError("--countdown/--rest 必须为有限非负数")
        if not 1 <= args.blocks_per_sm <= 8 or not 1 <= args.memory_mib <= 65536:
            raise ValueError("--blocks-per-sm 范围 1–8，--memory-mib 范围 1–65536，实际受空闲显存限制")
        if not 1 <= args.work_repeats <= 4096:
            raise ValueError("--work-repeats 范围 1–4096")
        if not 1 <= args.tensor_reuse <= 256:
            raise ValueError("--tensor-reuse 范围 1–256")
        if not math.isfinite(args.link_gib_s) or not 0 <= args.link_gib_s <= 16:
            raise ValueError("--link-gib-s 范围 0–16")
        if args.link_gib_s and len(devices) != 2:
            raise ValueError("双卡通信负载需要两张卡")
        if not 1 <= args.link_chunk_mib <= 64:
            raise ValueError("--link-chunk-mib 范围 1–64")
        if args.engine == "burst" and args.carrier not in ("tensor", "fp4"):
            raise ValueError("--engine burst 需要 --carrier tensor 或 fp4")
        segments = make_schedule(args, len(devices))
        total = sum(segment.seconds for segment in segments)
        print(f"GPU 活动调制：{args.carrier} / {args.engine}，高负载比例 {args.duty:g}，总时长 {total:.3f}s")
        for segment in segments[:12]:
            voices = " / ".join("休止" if value is None else f"{value:.2f} Hz" for value in segment.frequencies)
            print(f"  {segment.label:10s} {segment.seconds:.3f}s -> {voices}")
        if args.dry_run:
            return
        result = run(args, segments, devices)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"诊断时间表：{args.manifest}")
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
