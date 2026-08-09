#!/usr/bin/env python3
"""Suppress common fan noise while preserving workload-specific tonal lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import butter, istft, sosfiltfilt, stft


EPSILON = 1.0e-10


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float64) / 2147483648.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float64)
    else:
        raise SystemExit(f"不支持的 WAV 格式: {path} ({data.dtype})")
    if audio.ndim == 1:
        audio = audio[:, None]
    return sample_rate, audio


def transform(audio: np.ndarray, sample_rate: int, fft_size: int) -> tuple[np.ndarray, np.ndarray]:
    channels = []
    frequencies = None
    for channel in range(audio.shape[1]):
        frequencies, _, spectrum = stft(
            audio[:, channel],
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - fft_size // 8,
            boundary="zeros",
            padded=True,
        )
        channels.append(spectrum)
    assert frequencies is not None
    return frequencies, np.stack(channels)


def normalized_profile(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + EPSILON)
    profile = np.median(magnitude_db, axis=2)
    useful = (frequencies >= 100) & (frequencies <= 12_000)
    level = np.median(profile[:, useful], axis=1, keepdims=True)
    return profile - level


def build_common_profile(
    spectra: list[np.ndarray],
    frequencies: np.ndarray,
) -> np.ndarray:
    profiles = np.stack([normalized_profile(item, frequencies) for item in spectra])
    # A median across recordings rejects tones that belong to only one matrix size,
    # while retaining fan broadband shape and fixed blade/electrical tones.
    return np.median(profiles, axis=0)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def clean_spectrum(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    common_profile_db: np.ndarray,
    strength: float,
    floor_db: float,
) -> np.ndarray:
    magnitude = np.abs(spectrum)
    magnitude_db = 20.0 * np.log10(magnitude + EPSILON)
    useful = (frequencies >= 100) & (frequencies <= 12_000)
    frame_level = np.median(magnitude_db[:, useful, :], axis=1)
    estimated_noise_db = common_profile_db[:, :, None] + frame_level[:, None, :]

    # Positive values are features unique to this recording rather than common fan noise.
    unique_db = magnitude_db - estimated_noise_db
    unique_mask = sigmoid((unique_db - 1.5 * strength) / 1.8)

    # Tonal coil whine is narrow in frequency; fan/wind noise has a broad local envelope.
    local_envelope_db = median_filter(magnitude_db, size=(1, 61, 1), mode="nearest")
    tonal_db = magnitude_db - local_envelope_db
    tonal_mask = sigmoid((tonal_db - 3.0) / 1.5)

    floor = 10.0 ** (floor_db / 20.0)
    mask = unique_mask * (0.20 + 0.80 * tonal_mask)
    mask = floor + (1.0 - floor) * mask

    # Remove wind rumble and ultrasonic hiss with gentle transitions.
    high_pass = np.clip((frequencies - 70.0) / 100.0, 0.0, 1.0)
    low_pass = np.clip((14_000.0 - frequencies) / 2_000.0, 0.0, 1.0)
    mask *= high_pass[None, :, None] * low_pass[None, :, None]
    mask = gaussian_filter(mask, sigma=(0.0, 1.0, 0.8), mode="nearest")
    return spectrum * mask


def inverse_transform(
    spectrum: np.ndarray,
    sample_rate: int,
    fft_size: int,
    length: int,
) -> np.ndarray:
    channels = []
    for channel in range(spectrum.shape[0]):
        _, audio = istft(
            spectrum[channel],
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - fft_size // 8,
            input_onesided=True,
            boundary=True,
        )
        channels.append(audio[:length])
    result = np.stack(channels, axis=1)
    if result.shape[0] < length:
        result = np.pad(result, ((0, length - result.shape[0]), (0, 0)))
    return result


def high_pass(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    sos = butter(4, 65.0, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, audio, axis=0)


def peak_normalize(audio: np.ndarray, maximum_gain_db: float = 18.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < EPSILON:
        return audio
    target = 10.0 ** (-1.0 / 20.0)
    gain = min(target / peak, 10.0 ** (maximum_gain_db / 20.0))
    return audio * gain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="利用多段录音的共同频谱压制风噪，保留显卡窄带啸叫"
    )
    parser.add_argument("files", nargs="+", type=Path, help="两段或更多 WAV 效果最佳")
    parser.add_argument("--output-dir", type=Path, default=Path("denoised"))
    parser.add_argument("--strength", type=float, default=1.0, help="风噪抑制强度，默认 1.0")
    parser.add_argument("--floor-db", type=float, default=-30.0, help="最低保留增益，默认 -30 dB")
    parser.add_argument("--fft-size", type=int, default=8192, help="STFT 大小，默认 8192")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strength <= 0:
        raise SystemExit("--strength 必须大于 0")
    if args.floor_db > 0 or args.floor_db < -80:
        raise SystemExit("--floor-db 必须在 -80 到 0 之间")
    if args.fft_size < 1024 or args.fft_size & (args.fft_size - 1):
        raise SystemExit("--fft-size 必须是至少 1024 的 2 次幂")

    recordings = []
    spectra = []
    sample_rate = None
    frequencies = None
    channels = None
    for path in args.files:
        current_rate, audio = read_wav(path)
        if sample_rate is None:
            sample_rate = current_rate
            channels = audio.shape[1]
        if current_rate != sample_rate or audio.shape[1] != channels:
            raise SystemExit("所有录音必须具有相同采样率和声道数")
        current_frequencies, spectrum = transform(audio, current_rate, args.fft_size)
        recordings.append((path, audio))
        spectra.append(spectrum)
        frequencies = current_frequencies

    assert sample_rate is not None and frequencies is not None
    common_profile = build_common_profile(spectra, frequencies)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for (path, audio), spectrum in zip(recordings, spectra):
        cleaned_spectrum = clean_spectrum(
            spectrum,
            frequencies,
            common_profile,
            args.strength,
            args.floor_db,
        )
        cleaned = inverse_transform(cleaned_spectrum, sample_rate, args.fft_size, len(audio))
        cleaned = peak_normalize(high_pass(cleaned, sample_rate))
        output_path = args.output_dir / f"{path.stem}_denoised.wav"
        pcm = np.clip(cleaned * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, pcm)

        input_rms = 20.0 * np.log10(np.sqrt(np.mean(audio * audio)) + EPSILON)
        output_rms = 20.0 * np.log10(np.sqrt(np.mean(cleaned * cleaned)) + EPSILON)
        print(f"{path} -> {output_path} | RMS {input_rms:.1f} -> {output_rms:.1f} dBFS")


if __name__ == "__main__":
    main()
