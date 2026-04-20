from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class ProsodyFeatures:
    sample_rate: int
    hop_length: int
    win_length: int
    times_sec: np.ndarray
    f0_hz: np.ndarray
    voiced_mask: np.ndarray
    log_energy: np.ndarray

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            times_sec=self.times_sec,
            f0_hz=self.f0_hz,
            voiced_mask=self.voiced_mask.astype(np.int32),
            log_energy=self.log_energy,
        )

    def to_json_summary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_rate": int(self.sample_rate),
            "hop_length": int(self.hop_length),
            "win_length": int(self.win_length),
            "num_frames": int(self.times_sec.shape[0]),
            "voiced_ratio": float(self.voiced_mask.mean()),
            "f0_hz_mean_voiced": float(self.f0_hz[self.voiced_mask].mean()) if self.voiced_mask.any() else 0.0,
            "f0_hz_std_voiced": float(self.f0_hz[self.voiced_mask].std()) if self.voiced_mask.any() else 0.0,
            "log_energy_mean": float(self.log_energy.mean()),
            "log_energy_std": float(self.log_energy.std()),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_audio(path: Path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32), sr


def extract_f0(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float = 75.0,
    fmax: float = 400.0,
    method: str = "yin",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        voiced_mask = np.asarray(voiced_flag, dtype=bool)
        return f0, voiced_mask

    # Faster fallback for long files: use YIN and infer voiced regions from valid range.
    f0 = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float32)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced_mask = np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)
    f0 = np.where(voiced_mask, f0, 0.0).astype(np.float32)
    return f0, voiced_mask


def extract_log_energy(audio: np.ndarray, hop_length: int, win_length: int) -> np.ndarray:
    rms = librosa.feature.rms(y=audio, hop_length=hop_length, frame_length=win_length, center=True)[0]
    return np.log(np.maximum(rms, 1e-6)).astype(np.float32)


def extract_prosody(
    audio_path: Path,
    sample_rate: int = 16000,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 75.0,
    fmax: float = 400.0,
    pitch_method: str = "yin",
) -> ProsodyFeatures:
    audio, sr = load_audio(audio_path, sample_rate=sample_rate)
    f0_hz, voiced_mask = extract_f0(audio, sample_rate=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, method=pitch_method)
    log_energy = extract_log_energy(audio, hop_length=hop_length, win_length=win_length)
    n = min(len(f0_hz), len(log_energy))
    times_sec = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return ProsodyFeatures(
        sample_rate=sr,
        hop_length=hop_length,
        win_length=win_length,
        times_sec=times_sec.astype(np.float32),
        f0_hz=f0_hz[:n],
        voiced_mask=voiced_mask[:n],
        log_energy=log_energy[:n],
    )


def load_prosody_npz(path: Path) -> ProsodyFeatures:
    data = np.load(path)
    return ProsodyFeatures(
        sample_rate=int(data["sample_rate"]),
        hop_length=int(data["hop_length"]),
        win_length=int(data["win_length"]),
        times_sec=data["times_sec"].astype(np.float32),
        f0_hz=data["f0_hz"].astype(np.float32),
        voiced_mask=data["voiced_mask"].astype(bool),
        log_energy=data["log_energy"].astype(np.float32),
    )


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
