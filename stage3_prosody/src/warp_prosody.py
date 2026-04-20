from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from prosody import load_audio, load_prosody_npz, write_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warp query prosody toward reference via DTW path.")
    p.add_argument("--reference-npz", type=Path, required=True)
    p.add_argument("--query-npz", type=Path, required=True)
    p.add_argument("--dtw-npz", type=Path, required=True)
    p.add_argument("--query-audio", type=Path, required=True)
    p.add_argument("--out-audio", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--energy-mix", type=float, default=0.7)
    p.add_argument("--pitch-shift-limit-semitones", type=float, default=4.0)
    p.add_argument("--pitch-block-sec", type=float, default=0.12, help="Block size for time-varying pitch shift approximation.")
    p.add_argument("--pitch-smooth-sec", type=float, default=0.10, help="Smooth semitone targets over this window before blockwise shifting.")
    p.add_argument("--energy-smooth-sec", type=float, default=0.10, help="Smooth framewise gain envelope over this window.")
    p.add_argument("--time-warp-mix", type=float, default=0.35, help="Blend factor for DTW-derived local timing warp (0 disables, 1 follows reference timing shape).")
    return p.parse_args()


def build_query_to_ref_map(path: np.ndarray, query_len: int) -> np.ndarray:
    out = np.zeros(query_len, dtype=np.int32)
    buckets: list[list[int]] = [[] for _ in range(query_len)]
    for ref_idx, qry_idx in path:
        if 0 <= qry_idx < query_len:
            buckets[qry_idx].append(int(ref_idx))
    for i in range(query_len):
        if buckets[i]:
            out[i] = int(np.median(buckets[i]))
        else:
            out[i] = out[i - 1] if i > 0 else 0
    return out


def semitone_shift_from_ratio(ratio: float) -> float:
    return 12.0 * np.log2(max(ratio, 1e-6))


def smooth_curve(x: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    if win_frames % 2 == 0:
        win_frames += 1
    pad = win_frames // 2
    padded = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(win_frames, dtype=np.float32) / float(win_frames)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def monotonic_cummax(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    if out.size == 0:
        return out
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def apply_time_warp(
    audio: np.ndarray,
    sr: int,
    query_times_sec: np.ndarray,
    mapped_ref_times_sec: np.ndarray,
    mix: float,
) -> tuple[np.ndarray, dict]:
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-6 or len(query_times_sec) < 2:
        return audio.astype(np.float32), {"enabled": False, "mix": mix}

    query_end = float(query_times_sec[-1]) if float(query_times_sec[-1]) > 0 else (len(audio) - 1) / float(sr)
    ref_end = float(mapped_ref_times_sec[-1]) if float(mapped_ref_times_sec[-1]) > 0 else 1.0
    q_norm = query_times_sec / max(query_end, 1e-6)
    r_norm = mapped_ref_times_sec / max(ref_end, 1e-6)
    target_norm = (1.0 - mix) * q_norm + mix * r_norm
    target_norm = np.clip(monotonic_cummax(target_norm), 0.0, None)
    if target_norm[-1] <= 0:
        return audio.astype(np.float32), {"enabled": False, "mix": mix}
    target_norm = target_norm / target_norm[-1]
    target_times_sec = target_norm * query_end
    target_times_sec = monotonic_cummax(target_times_sec)

    sample_times = np.arange(len(audio), dtype=np.float32) / float(sr)
    source_times = np.interp(sample_times, target_times_sec, query_times_sec, left=query_times_sec[0], right=query_times_sec[-1])
    warped = np.interp(source_times, sample_times, audio).astype(np.float32)
    return warped, {
        "enabled": True,
        "mix": mix,
        "query_duration_sec": query_end,
        "target_end_sec": float(target_times_sec[-1]),
    }


def blockwise_pitch_shift(audio: np.ndarray, sr: int, n_steps_per_frame: np.ndarray, hop_length: int, block_sec: float) -> tuple[np.ndarray, dict]:
    """
    Approximate time-varying pitch shift by applying librosa pitch_shift on short blocks
    using the median target semitone shift for frames in that block, with simple crossfades.
    """
    block = max(0.05, float(block_sec))
    block_samples = int(round(block * sr))
    if block_samples <= 0:
        return audio, {"blocks": 0}

    frame_times = librosa.frames_to_time(np.arange(len(n_steps_per_frame)), sr=sr, hop_length=hop_length)
    # Map each sample to a frame index for quick aggregation.
    sample_times = np.arange(len(audio)) / float(sr)
    frame_idx_for_sample = np.searchsorted(frame_times, sample_times, side="right") - 1
    frame_idx_for_sample = np.clip(frame_idx_for_sample, 0, len(n_steps_per_frame) - 1)

    out = np.zeros_like(audio, dtype=np.float32)
    fade = int(0.02 * sr)  # 20ms crossfade
    pos = 0
    blocks = 0
    shifts = []
    while pos < len(audio):
        end = min(len(audio), pos + block_samples)
        idx0 = int(frame_idx_for_sample[pos])
        idx1 = int(frame_idx_for_sample[end - 1])
        block_shift = float(np.median(n_steps_per_frame[idx0 : idx1 + 1]))
        shifts.append(block_shift)
        target_len = end - pos
        y = librosa.effects.pitch_shift(audio[pos:end], sr=sr, n_steps=block_shift).astype(np.float32)
        # Librosa can occasionally return off-by-a-few samples; enforce exact length.
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
        elif y.shape[0] > target_len:
            y = y[:target_len]

        if blocks == 0:
            out[pos:end] = y
        else:
            # overlap-add with a short crossfade at block boundary.
            overlap = min(fade, target_len)
            if overlap > 0:
                w = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                out[pos : pos + overlap] = out[pos : pos + overlap] * (1.0 - w) + y[:overlap] * w
                out[pos + overlap : end] = y[overlap:]
            else:
                out[pos:end] = y

        pos = end
        blocks += 1

    meta = {
        "blocks": int(blocks),
        "block_sec": float(block),
        "fade_sec": float(fade / sr),
        "median_semitone_shift": float(np.median(shifts)) if shifts else 0.0,
        "mean_abs_semitone_shift": float(np.mean(np.abs(shifts))) if shifts else 0.0,
    }
    return out, meta


def main() -> None:
    args = parse_args()
    ref = load_prosody_npz(args.reference_npz)
    qry = load_prosody_npz(args.query_npz)
    dtw = np.load(args.dtw_npz)
    path = dtw["path"]

    q2r = build_query_to_ref_map(path, len(qry.times_sec))
    mapped_ref_f0 = ref.f0_hz[np.clip(q2r, 0, len(ref.f0_hz) - 1)]
    mapped_ref_energy = ref.log_energy[np.clip(q2r, 0, len(ref.log_energy) - 1)]

    qry_voiced = qry.voiced_mask & (qry.f0_hz > 1.0)
    ref_voiced = mapped_ref_f0 > 1.0
    voiced_overlap = qry_voiced & ref_voiced

    # Build a per-frame semitone target (voiced-only). We'll apply this approximately
    # via blockwise pitch shifting so DTW actually influences pitch over time.
    n_steps = np.zeros_like(qry.f0_hz, dtype=np.float32)
    if np.any(voiced_overlap):
        ratio = mapped_ref_f0 / np.maximum(qry.f0_hz, 1e-6)
        n_steps[voiced_overlap] = 12.0 * np.log2(np.maximum(ratio[voiced_overlap], 1e-6))
    n_steps = np.clip(n_steps, -args.pitch_shift_limit_semitones, args.pitch_shift_limit_semitones).astype(np.float32)
    pitch_smooth_frames = max(1, int(round(args.pitch_smooth_sec * qry.sample_rate / qry.hop_length)))
    n_steps = smooth_curve(n_steps, pitch_smooth_frames)

    audio, sr = load_audio(args.query_audio, sample_rate=qry.sample_rate)
    warped, pitch_meta = blockwise_pitch_shift(audio, sr=sr, n_steps_per_frame=n_steps, hop_length=qry.hop_length, block_sec=args.pitch_block_sec)

    frame_gain = np.exp(args.energy_mix * (mapped_ref_energy - qry.log_energy))
    energy_smooth_frames = max(1, int(round(args.energy_smooth_sec * qry.sample_rate / qry.hop_length)))
    frame_gain = smooth_curve(frame_gain.astype(np.float32), energy_smooth_frames)
    gain_times = librosa.frames_to_time(np.arange(len(frame_gain)), sr=sr, hop_length=qry.hop_length)
    sample_times = np.arange(len(warped)) / float(sr)
    interp_gain = np.interp(sample_times, gain_times, frame_gain, left=frame_gain[0], right=frame_gain[-1])
    warped = warped * interp_gain
    time_warped, time_meta = apply_time_warp(
        warped,
        sr=sr,
        query_times_sec=qry.times_sec,
        mapped_ref_times_sec=ref.times_sec[np.clip(q2r, 0, len(ref.times_sec) - 1)],
        mix=args.time_warp_mix,
    )
    warped = time_warped
    peak = np.max(np.abs(warped))
    if peak > 0.99:
        warped = warped * (0.99 / peak)

    write_audio(args.out_audio, warped.astype(np.float32), sr)
    payload = {
        "reference_npz": str(args.reference_npz),
        "query_npz": str(args.query_npz),
        "dtw_npz": str(args.dtw_npz),
        "query_audio": str(args.query_audio),
        "out_audio": str(args.out_audio),
        "pitch_block_sec": args.pitch_block_sec,
        "pitch_smooth_sec": args.pitch_smooth_sec,
        "pitch_meta": pitch_meta,
        "energy_mix": args.energy_mix,
        "energy_smooth_sec": args.energy_smooth_sec,
        "time_warp_mix": args.time_warp_mix,
        "time_warp_meta": time_meta,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"out_audio={args.out_audio}")
    print(f"out_json={args.out_json}")
    print(f"pitch_blocks={pitch_meta.get('blocks')}")


if __name__ == "__main__":
    main()
