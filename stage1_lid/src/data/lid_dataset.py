from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_audio_mono(path: str, target_sr: int) -> tuple[torch.Tensor, int]:
    """
    Robust loader for wav/flac/etc.
    Prefer torchaudio, fall back to soundfile (helps when torchaudio backend lacks FLAC support).
    Returns (waveform[1, T], sample_rate).
    """
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


class LIDFrameDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )
        self.groups = self._load_groups()
        if not self.groups:
            raise ValueError(f'No rows found for split={split} in {self.manifest_path}')

    def _load_groups(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        with self.manifest_path.open('r', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['split'] != self.split:
                    continue
                utt = row['utterance_id']
                entry = grouped.setdefault(
                    utt,
                    {
                        'audio_path': row['audio_path'],
                        'utterance_id': utt,
                        'source': row['source'],
                        'labels': [],
                        'confidence': [],
                    },
                )
                entry['labels'].append(int(row['label_id']))
                entry['confidence'].append(float(row['confidence']))
        return list(grouped.values())

    def compute_class_weights(self) -> torch.Tensor:
        counts = Counter()
        for item in self.groups:
            # Ignore unlabeled frames (they use the training ignore index).
            counts.update([lab for lab in item['labels'] if lab != -100])
        total = sum(counts.values())
        num_classes = max(counts) + 1
        weights = []
        for class_id in range(num_classes):
            count = counts[class_id]
            weights.append(total / (num_classes * max(count, 1)))
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.groups[index]
        waveform, _sr = load_audio_mono(item['audio_path'], self.sample_rate)
        features = self.mel(waveform).squeeze(0).transpose(0, 1)
        features = torch.log(features.clamp_min(1e-5))
        labels = torch.tensor(item['labels'], dtype=torch.long)
        confidence = torch.tensor(item['confidence'], dtype=torch.float32)

        frame_count = min(features.shape[0], labels.shape[0])
        features = features[:frame_count]
        labels = labels[:frame_count]
        confidence = confidence[:frame_count]

        return {
            'features': features,
            'labels': labels,
            'confidence': confidence,
            'length': frame_count,
            'utterance_id': item['utterance_id'],
            'source': item['source'],
        }


def lid_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    confidence = [item['confidence'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)

    feature_pad = pad_sequence(features, batch_first=True)
    label_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
    conf_pad = pad_sequence(confidence, batch_first=True, padding_value=0.0)
    mask = torch.arange(feature_pad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        'features': feature_pad,
        'labels': label_pad,
        'confidence': conf_pad,
        'lengths': lengths,
        'mask': mask,
        'utterance_ids': [item['utterance_id'] for item in batch],
        'sources': [item['source'] for item in batch],
    }
