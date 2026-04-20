from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class LFCCFrontend:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 64,
        n_lfcc: int = 20,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        self.sample_rate = sample_rate
        self.transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "center": True,
                "power": 2.0,
            },
            log_lf=True,
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feat = self.transform(wav).squeeze(0)  # [C, T]
        feat = feat - feat.mean(dim=1, keepdim=True)
        feat = feat / feat.std(dim=1, keepdim=True).clamp_min(1e-5)
        return feat


def load_audio_mono(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


class SpoofWindowDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        frontend: LFCCFrontend,
        num_frames: int = 200,
    ) -> None:
        self.rows = rows
        self.frontend = frontend
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.rows)

    def _fix_length(self, feat: torch.Tensor) -> torch.Tensor:
        cur = feat.size(1)
        if cur == self.num_frames:
            return feat
        if cur > self.num_frames:
            return feat[:, : self.num_frames]
        pad = self.num_frames - cur
        return torch.nn.functional.pad(feat, (0, pad))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        wav = load_audio_mono(row["audio_path"])
        start = int(float(row["start_sec"]) * self.frontend.sample_rate)
        end = int(float(row["end_sec"]) * self.frontend.sample_rate)
        clip = wav[start:end]
        feat = self.frontend(clip)
        feat = self._fix_length(feat).unsqueeze(0)  # [1, C, T]
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return feat, label


class LFCCCMNet(nn.Module):
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.head(x)


def eer_from_scores(y_true: np.ndarray, spoof_scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(spoof_scores).astype(np.float64)
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return {"eer": math.inf, "threshold": 0.0}
    thresholds = np.concatenate(([thresholds.min() - 1e-6], thresholds, [thresholds.max() + 1e-6]))

    best = None
    for thr in thresholds:
        pred_spoof = scores >= thr
        fa = ((pred_spoof == 1) & (y_true == 0)).sum()
        fr = ((pred_spoof == 0) & (y_true == 1)).sum()
        n_bona = max((y_true == 0).sum(), 1)
        n_spoof = max((y_true == 1).sum(), 1)
        far = fa / n_bona
        frr = fr / n_spoof
        diff = abs(far - frr)
        if best is None or diff < best[0]:
            best = (diff, (far + frr) / 2.0, thr, far, frr)
    assert best is not None
    return {
        "eer": float(best[1]),
        "threshold": float(best[2]),
        "far": float(best[3]),
        "frr": float(best[4]),
    }
