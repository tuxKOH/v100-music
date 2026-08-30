#!/usr/bin/env python3
"""Run two V100s in lockstep with identical FP32 GEMM workloads."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import wave
from collections.abc import Sequence
from pathlib import Path

import torch


SWEEP_SIZES = (512, 768, 1024, 1536, 2048)


def make_marker() -> str:
    rate, length = 44100, 0.16
    t = torch.arange(int(rate * length), dtype=torch.float64)
    samples = (0.45 * torch.sin(2 * torch.pi * 1760 * t / rate) *
               torch.hann_window(len(t), dtype=torch.float64)).numpy()
    path = tempfile.NamedTemporaryFile(prefix="v100_marker_", suffix=".wav", delete=False).name
    pcm = (samples * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def play_marker(path: str) -> float:
    started = time.time()
    for command in (("paplay", path), ("aplay", "-q", path)):
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return started
        except FileNotFoundError:
            continue
    print("警告：找不到 paplay/aplay，未播放 Line Out 定位音")
    return started


def countdown(seconds: float, label: str) -> None:
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        print(f"\r{label}: {remaining:4.1f}s ", end="", flush=True)
        time.sleep(min(0.1, remaining))
    print(f"\r{label}: GO!       ")


def parse_devices(value: str, count: int) -> list[torch.device]:
    try:
        indices = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--devices 格式应为 0,1") from exc
    if len(indices) != 2 or len(set(indices)) != 2:
        raise SystemExit("双卡齐走模式必须指定两张不同的 GPU，例如 --devices 0,1")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise SystemExit(f"CUDA 设备不存在: {invalid}；当前共有 {count} 张")
    return [torch.device(f"cuda:{index}") for index in indices]


def make_identical_matrices(
    size: int,
    devices: Sequence[torch.device],
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Generate once on the CPU so both cards receive bit-identical operands.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    host_a = torch.randn((size, size), dtype=torch.float32, generator=generator)
    host_b = torch.randn((size, size), dtype=torch.float32, generator=generator)
    matrices = []
    for device in devices:
        a = host_a.to(device)
        b = host_b.to(device)
        output = torch.empty_like(a)
        matrices.append((a, b, output))
    return matrices


def run_size(
    size: int,
    devices: Sequence[torch.device],
    duration: float,
    rest: float,
    repeats: int,
    seed: int,
    marker_path: str | None = None,
) -> dict:
    working_set_mib = 3 * size * size * 4 / 2**20
    print(
        f"\n=== FP32 {size}x{size} | {working_set_mib:.1f} MiB/card | "
        f"dual-GPU lockstep ==="
    )
    print("正在生成完全相同的矩阵并复制到两张卡……")
    matrices = make_identical_matrices(size, devices, seed)

    with torch.inference_mode():
        # Alternate submissions so neither card receives a long queue first.
        for _ in range(2):
            for a, b, output in matrices:
                torch.mm(a, b, out=output)
        for device in devices:
            torch.cuda.synchronize(device)

        marker_time = play_marker(marker_path) if marker_path else None
        countdown(rest, "静音/基线，请把手机放在两张卡中间")
        print(f"双卡 FP32 开始（持续 {duration:g} 秒）")
        started = time.monotonic()
        wall_started = time.time()
        batches = 0
        while time.monotonic() - started < duration:
            for _ in range(repeats):
                for a, b, output in matrices:
                    torch.mm(a, b, out=output)
            # A shared host-side barrier keeps each new batch aligned.
            for device in devices:
                torch.cuda.synchronize(device)
            batches += 1

    elapsed = time.monotonic() - started
    wall_ended = time.time()
    print(f"负载结束：双卡各 {batches * repeats} GEMMs / {elapsed:.2f}s")
    return {"size": size, "marker_unix": marker_time, "started_unix": wall_started, "ended_unix": wall_ended,
            "duration_s": elapsed, "batches_per_gpu": batches * repeats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="让两张 V100 使用相同数据同步运行纯 FP32 GEMM"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["fp32"],
        default="fp32",
        help="保留旧命令兼容；当前只有 fp32",
    )
    parser.add_argument("--fine-sweep", action="store_true",
                        help="512 到 2048 间按 --step 逐个测试，适合录音定位")
    parser.add_argument("--step", type=int, default=8,
                        help="--fine-sweep 的矩阵步长，默认 8")
    parser.add_argument("--manifest", type=Path,
                        help="把每段实际开始/结束时间写入 JSON")
    parser.add_argument("--lineout-marker", action="store_true",
                        help="每段开始前从电脑 Line Out 播放短定位音")
    parser.add_argument("--size", type=int, default=512, help="方阵边长，默认 512")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="依次测试 512,768,1024,1536,2048",
    )
    parser.add_argument("--duration", type=float, default=12.0, help="每个尺寸秒数，默认 12")
    parser.add_argument("--rest", type=float, default=5.0, help="每项开始前准备秒数，默认 5")
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="每次双卡同步前各自排队的 GEMM 数，默认 8",
    )
    parser.add_argument("--devices", default="0,1", help="两张 CUDA GPU，默认 0,1")
    parser.add_argument("--seed", type=int, default=100, help="相同矩阵的随机种子，默认 100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.size % 8 != 0:
        raise SystemExit("--size 必须是正数且为 8 的倍数")
    if args.duration <= 0 or args.rest < 0:
        raise SystemExit("--duration 必须大于 0，--rest 不能小于 0")
    if args.repeats <= 0:
        raise SystemExit("--repeats 必须大于 0")
    if args.step <= 0 or args.step % 8 != 0:
        raise SystemExit("--step 必须是正数且为 8 的倍数")
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA GPU；请在能看到 V100 的宿主机环境运行")

    devices = parse_devices(args.devices, torch.cuda.device_count())
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        print(
            f"{device}: {props.name}, {props.multi_processor_count} SM, "
            f"VRAM {props.total_memory / 2**30:.1f} GiB"
        )

    sizes = tuple(range(512, 2048 + 1, args.step)) if args.fine_sweep else (SWEEP_SIZES if args.sweep else (args.size,))
    manifest = []
    marker_path = make_marker() if args.lineout_marker else None
    try:
        for index, size in enumerate(sizes):
            print(f"\n扫描段 {index + 1}/{len(sizes)}：{size}x{size}")
            manifest.append(run_size(size, devices, args.duration, args.rest, args.repeats, args.seed, marker_path))
        if args.manifest:
            args.manifest.write_text(json.dumps({"duration_s": args.duration, "rest_s": args.rest,
                "step": args.step, "segments": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"定位清单已写入：{args.manifest}")
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
