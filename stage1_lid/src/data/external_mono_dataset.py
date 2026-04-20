from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from src.data.lid_dataset import load_audio_mono


LABEL_TO_ID = {"english": 0, "hindi": 1}


@dataclass(frozen=True)
class ExternalMonoItem:
    audio_path: str
    utterance_id: str
    split: str  # train|valid
    label_id: int
    source: str


def stable_split(key: str, valid_pct: int = 2) -> str:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "valid" if bucket < valid_pct else "train"


class ExternalMonoLIDDataset(torch.utils.data.Dataset):
    """
    Monolingual external dataset: derive per-frame labels on the fly.
    This avoids writing multi-GB frame-supervision CSVs.
    """

    def __init__(
        self,
        external_sources_csv: str | Path,
        split: str,
        *,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        max_items_per_lang: dict[str, int] | None = None,
        valid_pct: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.external_sources_csv = Path(external_sources_csv)
        self.split = split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )
        self.items = self._load_items(max_items_per_lang=max_items_per_lang, valid_pct=valid_pct)
        if not self.items:
            raise ValueError(f"No external mono items for split={split} in {self.external_sources_csv}")

    def _load_items(self, *, max_items_per_lang: dict[str, int] | None, valid_pct: int) -> list[ExternalMonoItem]:
        picked: dict[str, list[ExternalMonoItem]] = {"english": [], "hindi": []}
        with self.external_sources_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                lang = (row.get("language") or "").strip().lower()
                if lang not in ("english", "hindi"):
                    continue
                path = row["audio_path"]
                # Skip empty/broken files quickly
                try:
                    if Path(path).stat().st_size == 0:
                        continue
                except Exception:
                    continue

                row_split = (row.get("split") or "").strip().lower()
                if row_split == "valid":
                    s = "valid"
                else:
                    s = stable_split(path, valid_pct=valid_pct)
                if s != self.split:
                    continue

                # Cap per-language if requested.
                if max_items_per_lang and len(picked[lang]) >= max_items_per_lang.get(lang, 10**18):
                    continue

                utt = Path(path).stem
                picked[lang].append(
                    ExternalMonoItem(
                        audio_path=path,
                        utterance_id=utt,
                        split=s,
                        label_id=LABEL_TO_ID[lang],
                        source=f"{row.get('dataset','external')}_{lang}",
                    )
                )

        # Interleave to keep batches mixed.
        en = picked["english"]
        hi = picked["hindi"]
        out: list[ExternalMonoItem] = []
        i = 0
        while i < max(len(en), len(hi)):
            if i < len(en):
                out.append(en[i])
            if i < len(hi):
                out.append(hi[i])
            i += 1
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        waveform, _sr = load_audio_mono(item.audio_path, self.sample_rate)
        feats = self.mel(waveform).squeeze(0).transpose(0, 1)
        feats = torch.log(feats.clamp_min(1e-5))
        T = feats.shape[0]
        labels = torch.full((T,), int(item.label_id), dtype=torch.long)
        confidence = torch.ones((T,), dtype=torch.float32)
        return {
            "features": feats,
            "labels": labels,
            "confidence": confidence,
            "length": T,
            "utterance_id": item.utterance_id,
            "source": item.source,
        }

