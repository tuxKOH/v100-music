#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Learn a captured noise spectrum and preserve clearly observed tonal lines.

Reads PCM WAV files, including float WAV, and writes float32 PCM. CPU only;
does not use a musical score or synthesize replacement audio.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import istft, stft

FFT = 8192
HOP = 1024


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    rate, samples = wavfile.read(path)
    if samples.dtype.kind == "u":
        center = float(1 << (samples.dtype.itemsize * 8 - 1))
        samples = (samples.astype(np.float64) - center) / center
    elif samples.dtype.kind == "i":
        samples = samples.astype(np.float64) / float(1 << (samples.dtype.itemsize * 8 - 1))
    else:
        samples = samples.astype(np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError(f"{path}: WAV 必须包含有限的音频样本")
    return rate, samples


def transform(samples: np.ndarray, rate: int):
    spectra = []
    power = None
    for channel in range(samples.shape[1]):
        frequencies, times, z = stft(samples[:, channel], fs=rate, nperseg=FFT,
                                    noverlap=FFT-HOP, boundary="zeros", padded=True)
        spectra.append(z)
        p = np.abs(z)**2
        if power is None:
            power = p
        else:
            power += p
    return frequencies, times, spectra, power / samples.shape[1]


def learn_noise(reference: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) < 4 * FFT:
        raise ValueError("底噪片段太短；至少需要四个 FFT 窗口，建议录 60–120 秒")
    _, times, spectra, power = transform(reference, rate)
    valid = (times > FFT/rate) & (times < len(reference)/rate-FFT/rate)
    frame_db = 10*np.log10(np.sum(power, axis=0)+1e-25)
    # Exclude exceptionally loud handling/clipping events without discarding
    # ordinary pump fluctuations. Statistics are shared by all channels.
    valid &= frame_db < np.median(frame_db[valid])+18
    if valid.sum() < 4:
        raise ValueError("底噪片段没有足够的有效窗口")
    profile = np.mean([np.median(np.abs(z[:, valid])**2, axis=1)/np.log(2)
                       for z in spectra], axis=0)
    if np.max(profile) <= 1e-25:
        raise ValueError("底噪录音是静音，无法学习噪声频谱")
    tail = np.mean([np.percentile(np.abs(z[:, valid])**2, 90, axis=1)
                    for z in spectra], axis=0)
    variability = np.clip(10*np.log10((tail+1e-25)/
                                      (np.log(10)*profile+1e-25)), 0, 8)
    return profile, gaussian_filter1d(variability, sigma=.8)


def filter_block(samples: np.ndarray, rate: int, profile: np.ndarray,
                 variability: np.ndarray, variable_noise: bool) -> np.ndarray:
    frequencies, _, spectra, power = transform(samples, rate)
    useful = ((frequencies > 300) & (frequencies < min(7500, rate/2)) &
              (profile > profile.max()*1e-7))
    if not useful.any():
        useful = profile > profile.max()*1e-7
    level = np.median(power[useful]/(profile[useful, None]+1e-25), axis=0)/np.log(2)
    level = np.clip(gaussian_filter1d(level, sigma=.25*rate/HOP), .25, 4)
    noise = profile[:, None]*level[None, :]
    if variable_noise:
        # Match independent capture levels in broad bands. A small number of
        # musical lines cannot dominate these median level estimates.
        edges = [0, 120, 300, 1000, 4000, 8000, rate/2+1]
        edges = sorted({v for v in edges if v < rate/2} | {rate/2+1})
        for low, high in zip(edges[:-1], edges[1:]):
            bins = (frequencies >= low) & (frequencies < high)
            reliable = bins & (profile > profile.max()*1e-10)
            if reliable.any():
                band_level = np.median(power[reliable]/(profile[reliable, None]+1e-25), axis=0)/np.log(2)
                band_level = np.clip(gaussian_filter1d(band_level, sigma=.3*rate/HOP), .01, 100)
                noise[bins] = profile[bins, None]*band_level[None, :]
        noise = gaussian_filter(noise, sigma=(.5, 0))

    smooth_power = gaussian_filter(power, sigma=(.65, .9))
    snr_db = 10*np.log10((smooth_power+1e-25)/(noise+1e-25))
    power_db = 10*np.log10(smooth_power+1e-25)
    tonal_db = power_db-median_filter(power_db, size=(31, 1), mode="nearest")

    def sigmoid(value):
        return 1/(1+np.exp(-np.clip(value, -40, 40)))

    threshold = 4.5+.4*variability[:, None] if variable_noise else 5
    distinct = sigmoid((snr_db-threshold)/1.3)
    narrow = sigmoid((tonal_db-4)/1.2)
    floor = 10**((-32 if variable_noise else -26)/20)
    broad = .06 if variable_noise else .10
    gain = floor+(1-floor)*distinct*(broad+(1-broad)*narrow)
    gain = gaussian_filter(gain, sigma=(.55, .6))

    result = np.empty_like(samples)
    for channel, z in enumerate(spectra):
        _, cleaned = istft(z*gain, fs=rate, nperseg=FFT, noverlap=FFT-HOP,
                           input_onesided=True, boundary=True)
        result[:, channel] = cleaned[:len(samples)]
    if not np.all(np.isfinite(result)):
        raise ValueError("降噪结果包含无效样本")
    return result


def reduce_noise(samples: np.ndarray, rate: int, profile: np.ndarray,
                 variability: np.ndarray, variable_noise: bool = False,
                 chunk_seconds: float = 18, progress=None) -> np.ndarray:
    # Hop-aligned overlapping context keeps each block on the same STFT grid.
    # Context exceeds the temporal Gaussian filters' support plus an FFT.
    chunk = max(FFT, math.ceil(chunk_seconds*rate/HOP)*HOP)
    halo = max(2*FFT, math.ceil(2*rate/HOP)*HOP)
    result = np.empty_like(samples)
    for begin in range(0, len(samples), chunk):
        end = min(begin+chunk, len(samples))
        context_begin = max(0, begin-halo)
        context_end = min(len(samples), end+halo)
        block = filter_block(samples[context_begin:context_end], rate, profile,
                             variability, variable_noise)
        result[begin:end] = block[begin-context_begin:end-context_begin]
        if progress:
            progress(end/len(samples))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="用独立底噪 WAV 保留啸叫窄带音调并降噪")
    parser.add_argument("reference", type=Path, help="只有机器底噪的 WAV")
    parser.add_argument("recording", type=Path, help="要处理的演奏或第二段底噪 WAV")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--reference-start", type=float, default=0, help="噪声参考起点，秒")
    parser.add_argument("--reference-end", type=float, help="噪声参考终点，秒")
    parser.add_argument("--variable-noise", action="store_true", help="匹配各频段，并考虑水声等底噪的波动范围")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("输出已存在，请换个文件名；不覆盖原始音频")
        rate, reference = read_audio(args.reference)
        recording_rate, samples = read_audio(args.recording)
        if recording_rate != rate or reference.shape[1] != samples.shape[1]:
            raise ValueError("两段 WAV 的采样率和声道数必须一致，请先用 FFmpeg 转换")
        if len(samples) < FFT:
            raise ValueError("演奏录音太短，至少需要一个 FFT 窗口")
        end = len(reference)/rate if args.reference_end is None else args.reference_end
        if not (math.isfinite(args.reference_start) and math.isfinite(end) and
                0 <= args.reference_start < end <= len(reference)/rate):
            raise ValueError("噪声参考截取范围无效")
        reference = reference[int(args.reference_start*rate):int(end*rate)]
        print("学习底噪频谱……", flush=True)
        profile, variability = learn_noise(reference, rate)
        result = reduce_noise(samples, rate, profile, variability, args.variable_noise,
                              progress=lambda fraction: print(f"降噪 {fraction:.0%}", flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(args.output, rate, result.astype(np.float32))
        peak_db = 20*np.log10(np.max(np.abs(result))+1e-25)
        print(f"输出 {args.output}，时长 {len(result)/rate:.3f}s，峰值 {peak_db:.2f} dBFS")
        print("未自动提音量；合回视频时可使用固定增益，并检查输出峰值。")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
