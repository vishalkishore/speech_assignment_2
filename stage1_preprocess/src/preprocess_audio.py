from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Denoise + normalize a single audio file (spectral subtraction).")
    p.add_argument("input_audio", type=Path)
    p.add_argument("output_audio", type=Path)
    p.add_argument("--meta-json", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--noise-quantile", type=float, default=0.15)
    p.add_argument("--oversub", type=float, default=1.5)
    p.add_argument("--floor-ratio", type=float, default=0.08)
    p.add_argument("--smoothing", type=float, default=0.6)
    p.add_argument("--target-rms-dbfs", type=float, default=-24.0)
    p.add_argument("--peak-ceiling-dbfs", type=float, default=-1.0)
    return p.parse_args()


def to_mono_16k(path: Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).contiguous()


def rms_dbfs(wav: torch.Tensor) -> float:
    rms = torch.sqrt(torch.mean(torch.clamp(wav * wav, min=1e-12)))
    return float(20.0 * torch.log10(torch.clamp(rms, min=1e-12)))


def apply_rms_peak_normalization(wav: torch.Tensor, target_rms_dbfs: float, peak_ceiling_dbfs: float) -> tuple[torch.Tensor, dict]:
    cur_rms_db = rms_dbfs(wav)
    gain_db = target_rms_dbfs - cur_rms_db
    gain = float(10.0 ** (gain_db / 20.0))
    out = wav * gain

    peak = float(torch.max(torch.abs(out)).item())
    peak_ceiling = float(10.0 ** (peak_ceiling_dbfs / 20.0))
    peak_limiter_gain = 1.0
    if peak > peak_ceiling:
        peak_limiter_gain = peak_ceiling / max(peak, 1e-12)
        out = out * peak_limiter_gain

    return out, {
        "rms_dbfs_before": cur_rms_db,
        "gain_db_applied": gain_db,
        "peak_after_rms_gain": peak,
        "peak_limiter_gain": peak_limiter_gain,
        "rms_dbfs_after": rms_dbfs(out),
        "peak_after": float(torch.max(torch.abs(out)).item()),
    }


def estimate_noise_mag(mag: torch.Tensor, quantile: float) -> torch.Tensor:
    # mag shape: [freq, time]
    frame_energy = mag.mean(dim=0)  # [time]
    # If there are many exactly-silent frames, naive quantiles can pick only zeros and
    # make the noise estimate degenerate. Prefer the lowest-energy *non-zero* frames.
    nonzero = frame_energy > 0.0
    energy = frame_energy[nonzero] if int(nonzero.sum().item()) >= 2 else frame_energy

    thresh = torch.quantile(energy, q=quantile)
    mask = nonzero & (frame_energy <= thresh) if int(nonzero.sum().item()) >= 2 else (frame_energy <= thresh)

    if int(mask.sum().item()) < 2:
        k = max(2, int(0.1 * frame_energy.numel()))
        idx = torch.argsort(frame_energy)[:k]
        noise_mag = mag[:, idx].mean(dim=1, keepdim=True)
    else:
        noise_mag = mag[:, mask].mean(dim=1, keepdim=True)

    # Tiny floor so downstream stats are well-defined.
    return noise_mag.clamp_min(1e-8)


def smooth_over_time(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # x shape: [freq, time]
    y = x.clone()
    for t in range(1, x.size(1)):
        y[:, t] = alpha * y[:, t - 1] + (1.0 - alpha) * x[:, t]
    return y


def spectral_subtract(
    wav: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    noise_quantile: float,
    oversub: float,
    floor_ratio: float,
    smoothing: float,
) -> tuple[torch.Tensor, dict]:
    window = torch.hann_window(win_length, device=wav.device)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(spec)
    phase = torch.angle(spec)

    noise_mag = estimate_noise_mag(mag, noise_quantile)
    clean_mag = mag - oversub * noise_mag
    floor = floor_ratio * noise_mag
    clean_mag = torch.maximum(clean_mag, floor)
    clean_mag = smooth_over_time(clean_mag, alpha=smoothing)

    clean_spec = torch.polar(clean_mag, phase)
    denoised = torch.istft(
        clean_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=wav.numel(),
    )
    denoised = torch.clamp(denoised, -1.0, 1.0)

    noise_power = torch.mean(noise_mag * noise_mag).item()
    in_power = torch.mean(mag * mag).item()
    out_power = torch.mean(clean_mag * clean_mag).item()
    return denoised, {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "noise_quantile": noise_quantile,
        "oversub": oversub,
        "floor_ratio": floor_ratio,
        "smoothing": smoothing,
        "spec_in_power": in_power,
        "spec_noise_power_est": noise_power,
        "spec_out_power": out_power,
    }


def main() -> None:
    args = parse_args()
    args.output_audio.parent.mkdir(parents=True, exist_ok=True)
    if args.meta_json is not None:
        args.meta_json.parent.mkdir(parents=True, exist_ok=True)

    wav = to_mono_16k(args.input_audio, args.sample_rate)
    denoised, denoise_meta = spectral_subtract(
        wav=wav,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        noise_quantile=args.noise_quantile,
        oversub=args.oversub,
        floor_ratio=args.floor_ratio,
        smoothing=args.smoothing,
    )
    normalized, norm_meta = apply_rms_peak_normalization(
        denoised,
        target_rms_dbfs=args.target_rms_dbfs,
        peak_ceiling_dbfs=args.peak_ceiling_dbfs,
    )
    # Save as PCM16 for maximum downstream compatibility.
    wav_i16 = torch.clamp(normalized, -1.0, 1.0).mul(32767.0).round().to(torch.int16)
    torchaudio.save(str(args.output_audio), wav_i16.unsqueeze(0), args.sample_rate, encoding="PCM_S", bits_per_sample=16)

    meta = {
        "input_audio": str(args.input_audio),
        "output_audio": str(args.output_audio),
        "denoise": denoise_meta,
        "normalization": norm_meta,
    }
    if args.meta_json is not None:
        args.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote_audio={args.output_audio}")
    if args.meta_json is not None:
        print(f"wrote_meta={args.meta_json}")


if __name__ == "__main__":
    main()
