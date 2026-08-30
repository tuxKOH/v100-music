#!/usr/bin/env python3
"""Estimate stable tonal/coil-whine candidates from WAV recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, istft, stft


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        audio = data.astype(np.float64) / max(abs(info.min), info.max)
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV dtype: {data.dtype}")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return rate, audio


def analyse(path: Path, fft_size: int, min_hz: float, max_hz: float, top: int,
            extract_dir: Path | None = None) -> list[dict]:
    rate, audio = read_wav(path)
    freqs, _, z = stft(audio, fs=rate, window="hann", nperseg=fft_size,
                        noverlap=fft_size * 3 // 4, boundary="zeros", padded=True)
    db = 20.0 * np.log10(np.abs(z) + 1e-12)
    band = (freqs >= min_hz) & (freqs <= max_hz)
    band_db, band_freqs = db[band], freqs[band]
    median_db = np.median(band_db, axis=1)
    local = np.convolve(median_db, np.ones(9) / 9.0, mode="same")
    prominence = median_db - local
    distance = max(1, int(30 / (rate / fft_size)))
    peaks, _ = find_peaks(prominence, height=1.5, distance=distance)
    candidates = []
    frame_baseline = np.median(band_db, axis=0)
    for index in peaks:
        neighborhood = slice(max(0, index - 1), min(len(band_freqs), index + 2))
        local_peak = np.max(band_db[neighborhood], axis=0)
        occupancy = float(np.mean(local_peak > frame_baseline + 3.0))
        score = float(prominence[index] * (0.35 + occupancy))
        candidates.append({"frequency_hz": round(float(band_freqs[index]), 2),
                           "score": round(score, 3),
                           "prominence_db": round(float(prominence[index]), 2),
                           "occupancy": round(occupancy, 3)})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:top]
    if extract_dir is not None:
        # Reconstruct only narrow, persistent peaks.  A ~3-bin Gaussian keeps
        # the tone natural while rejecting broadband wind between the peaks.
        mask = np.zeros_like(z, dtype=np.float64)
        bin_hz = rate / fft_size
        for item in selected:
            center = int(round(item["frequency_hz"] / bin_hz))
            radius = max(1, int(round(35.0 / bin_hz)))
            indices = np.arange(max(0, center - radius), min(len(freqs), center + radius + 1))
            weights = np.exp(-0.5 * ((indices - center) / max(1.0, radius / 2.0)) ** 2)
            mask[indices, :] = np.maximum(mask[indices, :], weights[:, None])
        extracted = z * mask
        _, audio_out = istft(extracted, fs=rate, window="hann", nperseg=fft_size,
                             noverlap=fft_size * 3 // 4, input_onesided=True, boundary=True)
        audio_out = audio_out[: len(audio)]
        peak = float(np.max(np.abs(audio_out)))
        if peak > 1e-12:
            audio_out = audio_out * min(0.9 / peak, 10.0)
        pcm = np.clip(audio_out * 32767.0, -32768, 32767).astype(np.int16)
        extract_dir.mkdir(parents=True, exist_ok=True)
        output = extract_dir / f"{path.stem}_stable_whine.wav"
        wavfile.write(output, rate, pcm)
        print(f"稳定窄带提取: {output}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="纯计算提取稳定窄带啸叫候选")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--min-hz", type=float, default=100.0)
    parser.add_argument("--max-hz", type=float, default=12000.0)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--extract-dir", type=Path,
                        help="额外输出只含稳定窄带峰的 WAV 目录")
    args = parser.parse_args()
    if args.fft_size < 1024 or args.fft_size & (args.fft_size - 1):
        parser.error("--fft-size 必须是至少 1024 的 2 次幂")
    result = {str(path): analyse(path, args.fft_size, args.min_hz, args.max_hz, args.top,
                                 args.extract_dir)
              for path in args.files}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.json:
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
