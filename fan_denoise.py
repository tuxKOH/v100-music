#!/usr/bin/env python3
"""Reference-based fan-noise reduction that preserves narrow tonal peaks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import istft, stft

def read(path: Path):
    rate, data = wavfile.read(path)
    scale = float(np.iinfo(data.dtype).max) if np.issubdtype(data.dtype, np.integer) else 1.0
    audio = data.astype(np.float64) / scale
    if audio.ndim == 1: audio = audio[:, None]
    return rate, audio

def clean(path: Path, fan: np.ndarray, rate: int, fft: int, strength: float, out: Path):
    _, audio = read(path)
    if len(fan.shape) != 2: fan = fan[:, None]
    channels = min(audio.shape[1], fan.shape[1])
    result = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        ref = fan[:, min(ch, channels - 1)]
        _, _, zn = stft(ref, fs=rate, window='hann', nperseg=fft, noverlap=fft*3//4, boundary='zeros', padded=True)
        freqs, _, z = stft(audio[:, ch], fs=rate, window='hann', nperseg=fft, noverlap=fft*3//4, boundary='zeros', padded=True)
        noise = np.median(np.abs(zn), axis=1)[:, None]
        power = np.abs(z)
        # Match reference level per frame using the quiet lower half of bins.
        scale_frame = np.median(power / (noise + 1e-10), axis=0, keepdims=True)
        estimate = noise * np.clip(scale_frame, 0.35, 2.5) * strength
        ratio = np.maximum(power - estimate, 0.0) / (power + 1e-10)
        # Never erase sharp tonal lines: retain at least -6 dB at local peaks.
        local = np.median(20*np.log10(power + 1e-10), axis=1, keepdims=True)
        tonal = 20*np.log10(power + 1e-10) - local
        ratio = np.maximum(ratio, np.where(tonal > 5.0, 0.5, 0.0))
        ratio = np.maximum(ratio, 10**(-30/20))
        _, cleaned = istft(z * ratio, fs=rate, window='hann', nperseg=fft, noverlap=fft*3//4, input_onesided=True, boundary=True)
        result[:, ch] = cleaned[:len(audio)]
    peak = np.max(np.abs(result))
    if peak > .95: result *= .95 / peak
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, rate, np.clip(result*32767, -32768, 32767).astype(np.int16))
    print(f'{path} -> {out}')

def main():
    p=argparse.ArgumentParser(description='用只有风扇的录音做保守谱减')
    p.add_argument('fan_reference', type=Path)
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('fan_denoised'))
    p.add_argument('--strength', type=float, default=0.85)
    p.add_argument('--fft-size', type=int, default=16384)
    a=p.parse_args()
    rr, fan=read(a.fan_reference)
    if a.strength <= 0: p.error('--strength 必须大于 0')
    for path in a.files:
        rate, _ = read(path)
        if rate != rr: p.error('fan.wav 与输入录音采样率必须一致')
        clean(path, fan, rr, a.fft_size, a.strength, a.output_dir/f'{path.stem}_fan_reduced.wav')
if __name__ == '__main__': main()
