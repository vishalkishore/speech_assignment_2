from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]  # assignment2-final/
DEFAULT_OUT_DIR = DEFAULT_PROJECT / "stage3_voice/embeddings"
DEFAULT_HF_CACHE = Path.home() / ".cache/huggingface/hub"
DEFAULT_SHUNYA_MODEL = "shunyalabs/zero-stt-hinglish"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a high-dimensional speaker embedding from a 60s reference clip.")
    parser.add_argument("input_audio", type=Path)
    parser.add_argument("--embedding-npy", type=Path, default=None)
    parser.add_argument("--meta-json", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prefer-backend", choices=["speechbrain", "torchaudio"], default="speechbrain")
    parser.add_argument("--speechbrain-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--offline-whisper-model", default=DEFAULT_SHUNYA_MODEL)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    return parser.parse_args()


def load_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    load_path = path
    tmp_path: Path | None = None
    if path.suffix.lower() in {".m4a", ".mp4", ".aac"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cmd = [
            "/snap/bin/ffmpeg",
            "-y",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr}")
        load_path = tmp_path

    waveform, sr = torchaudio.load(str(load_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if tmp_path is not None and tmp_path.exists():
        tmp_path.unlink()
    return waveform, sr


def duration_sec(waveform: torch.Tensor, sample_rate: int) -> float:
    return waveform.size(-1) / float(sample_rate)


def extract_with_speechbrain(
    waveform: torch.Tensor,
    sample_rate: int,
    source: str,
    hf_cache: Path,
    device: str,
) -> tuple[torch.Tensor, dict]:
    from speechbrain.pretrained import EncoderClassifier

    model_cache_id = source.replace("/", "__").replace(":", "_")
    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=str(hf_cache / model_cache_id),
        run_opts={"device": device},
    )
    emb = classifier.encode_batch(waveform)
    emb = emb.squeeze(0).squeeze(0).detach().cpu()
    return emb, {
        "backend": "speechbrain",
        "model_source": source,
        "embedding_dim": int(emb.numel()),
    }


def extract_with_torchaudio(
    waveform: torch.Tensor,
    sample_rate: int,
    device: str,
) -> tuple[torch.Tensor, dict]:
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    model = bundle.get_model().to(device)
    model.eval()
    wav = waveform.to(device)
    with torch.no_grad():
        feats, _ = model.extract_features(wav)
    stacked = torch.stack([f.squeeze(0) for f in feats], dim=0)
    pooled = stacked.mean(dim=1).mean(dim=0).detach().cpu()
    return pooled, {
        "backend": "torchaudio_wav2vec2",
        "model_source": "torchaudio.pipelines.WAV2VEC2_XLSR53",
        "embedding_dim": int(pooled.numel()),
    }


def extract_with_offline_whisper(
    waveform: torch.Tensor,
    model_id: str,
    hf_cache: Path,
    device: str,
) -> tuple[torch.Tensor, dict]:
    from transformers import WhisperModel, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        model_id,
        cache_dir=str(hf_cache),
        local_files_only=True,
    )
    model = WhisperModel.from_pretrained(
        model_id,
        cache_dir=str(hf_cache),
        local_files_only=True,
    ).to(device)
    model.eval()

    features = processor.feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)

    with torch.no_grad():
        enc = model.encoder(features).last_hidden_state
    pooled = enc.mean(dim=1).squeeze(0).detach().cpu()
    return pooled, {
        "backend": "offline_whisper_encoder",
        "model_source": model_id,
        "embedding_dim": int(pooled.numel()),
    }


def save_embedding(embedding: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embedding": embedding}, out_path)


def main() -> None:
    args = parse_args()
    stem = args.input_audio.stem
    embedding_path = args.embedding_npy or (DEFAULT_OUT_DIR / f"{stem}.pt")
    meta_path = args.meta_json or (DEFAULT_OUT_DIR / f"{stem}.json")

    waveform, sr = load_audio_16k(args.input_audio)
    clip_duration = duration_sec(waveform, sr)
    if clip_duration < 55.0 or clip_duration > 65.0:
        print(f"warning=expected_about_60_seconds got={clip_duration:.2f}s")

    backends = ["speechbrain", "torchaudio", "offline_whisper"] if args.prefer_backend == "speechbrain" else ["torchaudio", "speechbrain", "offline_whisper"]
    embedding = None
    meta = None
    last_error = None

    for backend in backends:
        try:
            if backend == "speechbrain":
                embedding, meta = extract_with_speechbrain(
                    waveform=waveform,
                    sample_rate=sr,
                    source=args.speechbrain_source,
                    hf_cache=args.hf_cache,
                    device=args.device,
                )
            else:
                if backend == "torchaudio":
                    embedding, meta = extract_with_torchaudio(
                        waveform=waveform,
                        sample_rate=sr,
                        device=args.device,
                    )
                else:
                    embedding, meta = extract_with_offline_whisper(
                        waveform=waveform,
                        model_id=args.offline_whisper_model,
                        hf_cache=args.hf_cache,
                        device=args.device,
                    )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{backend}:{exc}"

    if embedding is None or meta is None:
        raise RuntimeError(f"Failed to extract speaker embedding. last_error={last_error}")

    save_embedding(embedding, embedding_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update(
        {
            "input_audio": str(args.input_audio),
            "embedding_path": str(embedding_path),
            "sample_rate": sr,
            "duration_sec": clip_duration,
            "l2_norm": float(torch.linalg.norm(embedding).item()),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"embedding_path={embedding_path}")
    print(f"meta_path={meta_path}")
    print(f"backend={meta['backend']}")
    print(f"embedding_dim={meta['embedding_dim']}")


if __name__ == "__main__":
    main()
