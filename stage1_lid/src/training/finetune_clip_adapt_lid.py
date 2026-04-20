from __future__ import annotations

import argparse
import json
import time
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from src.data.external_mono_dataset import ExternalMonoLIDDataset
from src.data.lid_dataset import LIDFrameDataset, lid_collate_fn
from src.eval.metrics import frame_f1_metrics
from src.models.lid_bilstm import FrameLIDBiLSTM


PROJECT_ROOT = Path("/home/vishal/assignment2-final/stage1_lid")
DEFAULT_EXTERNAL_SOURCES = Path("/home/vishal/assignment2-final/data/manifests/shared/external_sources.csv")
DEFAULT_CLIP_MANIFEST = PROJECT_ROOT / "manifests/clip_pseudo_frame_supervision.csv"
DEFAULT_INIT = PROJECT_ROOT / "checkpoints/lid_external_mono_v2_overnight_best.pt"
DEFAULT_OUTDIR = PROJECT_ROOT / "checkpoints/lid_clip_adapt_clean"
DEFAULT_CLIP_UTTERANCES = PROJECT_ROOT / "manifests/utterances.csv"
DEFAULT_CLIP_SPANS = PROJECT_ROOT / "manifests/gemini_lid_spans_original.csv"
DEFAULT_SPLIT_UTTERANCES = PROJECT_ROOT / "manifests/utterances_clean_split.csv"
DEFAULT_SPLIT_SPANS = PROJECT_ROOT / "manifests/spans_clean_split.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LID on real clip pseudo-labels plus monolingual support data.")
    p.add_argument("--clip-manifest", type=Path, default=DEFAULT_CLIP_MANIFEST)
    p.add_argument("--external-sources", type=Path, default=DEFAULT_EXTERNAL_SOURCES)
    p.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--valid-pct", type=int, default=2)
    p.add_argument("--max-train-per-lang", type=int, default=2000)
    p.add_argument("--max-valid-per-lang", type=int, default=200)
    p.add_argument("--clip-utterances-csv", type=Path, default=DEFAULT_CLIP_UTTERANCES)
    p.add_argument("--clip-spans-csv", type=Path, default=DEFAULT_CLIP_SPANS)
    p.add_argument("--eval-clean-split", default="heldout_test")
    p.add_argument("--split-utterances-csv", type=Path, default=DEFAULT_SPLIT_UTTERANCES)
    p.add_argument("--split-spans-csv", type=Path, default=DEFAULT_SPLIT_SPANS)
    return p.parse_args()


def run_epoch(model, loader, optimizer, criterion, device, train: bool, log_every: int, epoch: int, total_epochs: int):
    model.train(train)
    total_loss = 0.0
    all_logits = []
    all_labels = []
    phase = "train" if train else "valid"
    epoch_start = time.time()
    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        confidence = batch["confidence"].to(device)
        logits = model(features)
        loss_raw = criterion(logits.transpose(1, 2), labels)
        valid = labels != -100
        loss = (loss_raw * confidence)[valid].mean() if valid.any() else loss_raw.mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu().reshape(-1, logits.size(-1)))
        all_labels.append(labels.detach().cpu().reshape(-1))
        if step % log_every == 0 or step == len(loader):
            elapsed = time.time() - epoch_start
            avg_step = elapsed / step
            eta = avg_step * (len(loader) - step)
            print(f"[{phase}] epoch {epoch}/{total_epochs} step {step}/{len(loader)} loss={total_loss/step:.4f} elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m", flush=True)
    metrics = frame_f1_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def evaluate_reference_clip(
    model,
    *,
    utterances_csv: Path,
    spans_csv: Path,
    device: torch.device,
    target_clean_split: str | None = None,
) -> dict[str, float]:
    hop_length = 160
    sample_rate = 16000
    hop_sec = hop_length / sample_rate
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=hop_length, n_mels=80, center=False, power=2.0)
    utt_rows = list(csv.DictReader(utterances_csv.open()))
    if target_clean_split is not None:
        utt_rows = [r for r in utt_rows if r.get("clean_split", "") == target_clean_split]
    utt_by_id = {r["utterance_id"]: r for r in utt_rows}
    spans_by_utt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv.DictReader(spans_csv.open()):
        if target_clean_split is not None and row.get("clean_split", "") != target_clean_split:
            continue
        spans_by_utt[row["utterance_id"]].append(row)
    all_logits = []
    all_labels = []
    utterances_used = 0
    model.eval()
    with torch.no_grad():
        for utt_id, spans in spans_by_utt.items():
            if utt_id not in utt_by_id:
                continue
            wav, sr = torchaudio.load(utt_by_id[utt_id]["audio_path"])
            wav = wav.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            feats = mel(wav).squeeze(0).transpose(0, 1)
            feats = torch.log(feats.clamp_min(1e-5))
            logits = model(feats.unsqueeze(0).to(device)).squeeze(0).cpu()
            T = logits.shape[0]
            labels = torch.full((T,), -100, dtype=torch.long)
            for span in spans:
                lang = span["language"].strip().lower()
                if lang not in {"english", "hindi"}:
                    continue
                label_id = 0 if lang == "english" else 1
                start = max(0, int(round(float(span["start_sec"]) / hop_sec)))
                end = min(T, int(round(float(span["end_sec"]) / hop_sec)))
                if end <= start:
                    end = min(T, start + 1)
                labels[start:end] = label_id
            valid = labels != -100
            if valid.any():
                all_logits.append(logits[valid])
                all_labels.append(labels[valid])
                utterances_used += 1
    if not all_logits:
        return {
            "english_f1": 0.0,
            "hindi_f1": 0.0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "utterances_used": 0,
        }
    metrics = frame_f1_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
    metrics["utterances_used"] = utterances_used
    return metrics


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    clip_train = LIDFrameDataset(args.clip_manifest, split="train")
    clip_valid = LIDFrameDataset(args.clip_manifest, split="valid")
    mono_train = ExternalMonoLIDDataset(args.external_sources, split="train", max_items_per_lang={"english": args.max_train_per_lang, "hindi": args.max_train_per_lang}, valid_pct=args.valid_pct)
    mono_valid = ExternalMonoLIDDataset(args.external_sources, split="valid", max_items_per_lang={"english": args.max_valid_per_lang, "hindi": args.max_valid_per_lang}, valid_pct=args.valid_pct)

    train_ds = ConcatDataset([clip_train, mono_train])
    valid_ds = ConcatDataset([clip_valid, mono_valid])

    print(f"device={args.device}", flush=True)
    print(f"clip_train={len(clip_train)} clip_valid={len(clip_valid)} mono_train={len(mono_train)} mono_valid={len(mono_valid)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lid_collate_fn, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lid_collate_fn, pin_memory=True)

    device = torch.device(args.device)
    model = FrameLIDBiLSTM().to(device)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_clip = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"starting_epoch={epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, True, args.log_every, epoch, args.epochs)
        valid_metrics = run_epoch(model, valid_loader, optimizer, criterion, device, False, args.log_every, epoch, args.epochs)
        clip_metrics = evaluate_reference_clip(
            model,
            utterances_csv=args.split_utterances_csv,
            spans_csv=args.split_spans_csv,
            device=device,
            target_clean_split=args.eval_clean_split,
        )
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "clip_eval": clip_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        payload = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "train_metrics": train_metrics, "valid_metrics": valid_metrics, "clip_eval": clip_metrics, "args": vars(args)}
        torch.save(payload, args.outdir / "last.pt")
        if clip_metrics["macro_f1"] > best_clip:
            best_clip = clip_metrics["macro_f1"]
            torch.save(payload, args.outdir / "best_clip.pt")

    (args.outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_clip_macro_f1={best_clip:.4f}", flush=True)
    print(f"checkpoints={args.outdir}", flush=True)


if __name__ == "__main__":
    main()
