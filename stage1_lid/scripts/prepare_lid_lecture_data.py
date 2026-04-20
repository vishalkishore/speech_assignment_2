from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path('/home/vishal/assignment2-final/stage1_lid')
LECTURE_MANIFEST = PROJECT_ROOT / 'manifests/lecture_sources.csv'
OUTPUT_AUDIO_DIR = PROJECT_ROOT / 'audio/chunks'
UTTERANCE_MANIFEST = PROJECT_ROOT / 'manifests/utterances.csv'
FRAME_LABEL_MANIFEST = PROJECT_ROOT / 'manifests/frame_labels_scaffold.csv'


@dataclass
class Segment:
    audio_path: Path
    utterance_id: str
    start_sec: float
    end_sec: float
    split: str
    transcript: str
    source: str
    num_samples: int
    sample_rate: int
    rms: float
    peak: float


def load_lecture_rows() -> list[dict[str, str]]:
    with LECTURE_MANIFEST.open('r', newline='') as handle:
        return list(csv.DictReader(handle))


def assign_split(index: int) -> str:
    if index == 0:
        return 'test'
    if index == 1:
        return 'valid'
    return 'train'


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    return audio.astype(np.float32), sample_rate


def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_sec: float,
    overlap_sec: float,
    min_chunk_sec: float,
) -> Iterable[tuple[int, int, np.ndarray]]:
    total_samples = waveform.shape[0]
    chunk_samples = int(chunk_sec * sample_rate)
    hop_samples = max(1, int((chunk_sec - overlap_sec) * sample_rate))
    min_chunk_samples = int(min_chunk_sec * sample_rate)

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        current = waveform[start:end]
        if current.shape[0] < min_chunk_samples:
            break
        yield start, end, current
        if end == total_samples:
            break
        start += hop_samples


def save_segment(segment_waveform: np.ndarray, sample_rate: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), segment_waveform, sample_rate)


def write_utterance_manifest(segments: list[Segment]) -> None:
    with UTTERANCE_MANIFEST.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'audio_path',
            'utterance_id',
            'start_sec',
            'end_sec',
            'split',
            'transcript',
            'source',
            'num_samples',
            'sample_rate',
            'rms',
            'peak',
        ])
        for seg in segments:
            writer.writerow([
                str(seg.audio_path),
                seg.utterance_id,
                f'{seg.start_sec:.3f}',
                f'{seg.end_sec:.3f}',
                seg.split,
                seg.transcript,
                seg.source,
                seg.num_samples,
                seg.sample_rate,
                f'{seg.rms:.6f}',
                f'{seg.peak:.6f}',
            ])


def write_frame_scaffold(segments: list[Segment], frame_ms: int, hop_ms: int) -> None:
    with FRAME_LABEL_MANIFEST.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'audio_path',
            'utterance_id',
            'frame_index',
            'frame_start_sec',
            'frame_end_sec',
            'label',
            'label_source',
            'confidence',
        ])
        for seg in segments:
            frame_len = frame_ms / 1000.0
            hop = hop_ms / 1000.0
            seg_dur = seg.end_sec - seg.start_sec
            frame_index = 0
            cur = 0.0
            while cur + frame_len <= seg_dur + 1e-9:
                writer.writerow([
                    str(seg.audio_path),
                    seg.utterance_id,
                    frame_index,
                    f'{cur:.3f}',
                    f'{cur + frame_len:.3f}',
                    '',
                    'unlabeled',
                    '',
                ])
                frame_index += 1
                cur += hop


def build_segments(args: argparse.Namespace) -> list[Segment]:
    rows = load_lecture_rows()
    segments: list[Segment] = []

    for lecture_index, row in enumerate(rows):
        lecture_path = Path(row['audio_path'])
        source_id = row['source_id']
        split = assign_split(lecture_index)
        waveform, sample_rate = read_audio(lecture_path)

        for chunk_idx, (start, end, chunk) in enumerate(
            chunk_waveform(
                waveform,
                sample_rate,
                args.chunk_sec,
                args.overlap_sec,
                args.min_chunk_sec,
            )
        ):
            utterance_id = f'{source_id}__chunk_{chunk_idx:04d}'
            out_path = OUTPUT_AUDIO_DIR / f'{utterance_id}.wav'
            save_segment(chunk, sample_rate, out_path)
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            peak = float(np.max(np.abs(chunk)))
            segments.append(
                Segment(
                    audio_path=out_path,
                    utterance_id=utterance_id,
                    start_sec=start / sample_rate,
                    end_sec=end / sample_rate,
                    split=split,
                    transcript='',
                    source=source_id,
                    num_samples=int(chunk.shape[0]),
                    sample_rate=sample_rate,
                    rms=rms,
                    peak=peak,
                )
            )
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare lecture chunks and frame scaffolds for LID.')
    parser.add_argument('--chunk-sec', type=float, default=15.0)
    parser.add_argument('--overlap-sec', type=float, default=1.0)
    parser.add_argument('--min-chunk-sec', type=float, default=5.0)
    parser.add_argument('--frame-ms', type=int, default=25)
    parser.add_argument('--hop-ms', type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    segments = build_segments(args)
    write_utterance_manifest(segments)
    write_frame_scaffold(segments, args.frame_ms, args.hop_ms)
    print(f'Prepared {len(segments)} lecture chunks')
    print(f'Chunk audio dir: {OUTPUT_AUDIO_DIR}')
    print(f'Utterance manifest: {UTTERANCE_MANIFEST}')
    print(f'Frame scaffold: {FRAME_LABEL_MANIFEST}')


if __name__ == '__main__':
    main()
