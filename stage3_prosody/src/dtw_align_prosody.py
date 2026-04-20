from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np

from prosody import load_prosody_npz


def zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def voiced_f0_feature(f0_hz: np.ndarray, voiced_mask: np.ndarray) -> np.ndarray:
    f0 = np.where(voiced_mask, f0_hz, 0.0)
    voiced_vals = f0[voiced_mask]
    if voiced_vals.size == 0:
        return np.zeros_like(f0)
    f0_log = np.zeros_like(f0)
    f0_log[voiced_mask] = np.log(np.maximum(voiced_vals, 1e-6))
    return f0_log


def build_feature_matrix(npz_path: Path) -> np.ndarray:
    p = load_prosody_npz(npz_path)
    f0_feat = zscore(voiced_f0_feature(p.f0_hz, p.voiced_mask))
    energy_feat = zscore(p.log_energy)
    vuv_feat = p.voiced_mask.astype(np.float32) * 2.0 - 1.0
    return np.stack([f0_feat, energy_feat, vuv_feat], axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DTW-align two prosody contour files.")
    p.add_argument("--reference-npz", type=Path, required=True)
    p.add_argument("--query-npz", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--frame-step", type=int, default=5, help="Decimate feature frames before DTW for tractable long-form alignment.")
    p.add_argument("--band-ratio", type=float, default=0.15, help="Sakoe-Chiba band as a fraction of the longer sequence length.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref = build_feature_matrix(args.reference_npz)
    qry = build_feature_matrix(args.query_npz)
    ref_len = ref.shape[1]
    qry_len = qry.shape[1]

    step = max(1, int(args.frame_step))
    ref_ds = ref[:, ::step]
    qry_ds = qry[:, ::step]
    band_rad = max(1, int(args.band_ratio * max(ref_ds.shape[1], qry_ds.shape[1])))

    cost, path = librosa.sequence.dtw(X=ref_ds, Y=qry_ds, metric="euclidean", band_rad=band_rad)
    path = np.asarray(path[::-1], dtype=np.int32)
    path = path * step
    path[:, 0] = np.clip(path[:, 0], 0, ref_len - 1)
    path[:, 1] = np.clip(path[:, 1], 0, qry_len - 1)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, cost=cost, path=path)

    payload = {
        "reference_npz": str(args.reference_npz),
        "query_npz": str(args.query_npz),
        "path_length": int(path.shape[0]),
        "final_cost": float(cost[-1, -1]),
        "reference_frames": int(ref_len),
        "query_frames": int(qry_len),
        "frame_step": int(step),
        "reference_frames_downsampled": int(ref_ds.shape[1]),
        "query_frames_downsampled": int(qry_ds.shape[1]),
        "band_ratio": float(args.band_ratio),
        "band_radius_frames": int(band_rad),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"out_npz={args.out_npz}")
    print(f"out_json={args.out_json}")
    print(f"path_length={payload['path_length']}")


if __name__ == "__main__":
    main()
