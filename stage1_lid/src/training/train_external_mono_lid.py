from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.external_mono_dataset import ExternalMonoLIDDataset
from src.data.lid_dataset import lid_collate_fn
from src.eval.metrics import frame_f1_metrics
from src.models.lid_bilstm import FrameLIDBiLSTM


PROJECT_ROOT = Path("/home/vishal/assignment2-final/stage1_lid")
DEFAULT_EXTERNAL_SOURCES = Path("/home/vishal/assignment2-final/data/manifests/shared/external_sources.csv")
DEFAULT_OUTDIR = PROJECT_ROOT / "checkpoints/lid_external_mono"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train frame-level LID on external monolingual data (EN+HI).")
    p.add_argument("--external-sources", type=Path, default=DEFAULT_EXTERNAL_SOURCES)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--valid-pct", type=int, default=2)
    p.add_argument("--max-train-per-lang", type=int, default=4000)
    p.add_argument("--max-valid-per-lang", type=int, default=400)
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
            print(
                f"[{phase}] epoch {epoch}/{total_epochs} "
                f"step {step}/{len(loader)} "
                f"loss={total_loss / step:.4f} "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )

    metrics = frame_f1_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_ds = ExternalMonoLIDDataset(
        args.external_sources,
        split="train",
        max_items_per_lang={"english": args.max_train_per_lang, "hindi": args.max_train_per_lang},
        valid_pct=args.valid_pct,
    )
    valid_ds = ExternalMonoLIDDataset(
        args.external_sources,
        split="valid",
        max_items_per_lang={"english": args.max_valid_per_lang, "hindi": args.max_valid_per_lang},
        valid_pct=args.valid_pct,
    )

    print(f"device={args.device}", flush=True)
    print(f"external_sources={args.external_sources}", flush=True)
    print(f"train_items={len(train_ds)} valid_items={len(valid_ds)}", flush=True)
    print(f"batch_size={args.batch_size}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lid_collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lid_collate_fn,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = FrameLIDBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"starting_epoch={epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, True, args.log_every, epoch, args.epochs)
        valid_metrics = run_epoch(model, valid_loader, optimizer, criterion, device, False, args.log_every, epoch, args.epochs)

        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(record)
        print(json.dumps(record), flush=True)

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "args": vars(args),
        }
        torch.save(payload, args.outdir / "last.pt")
        if valid_metrics["macro_f1"] > best:
            best = valid_metrics["macro_f1"]
            torch.save(payload, args.outdir / "best.pt")

    (args.outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best_valid_macro_f1={best:.4f}", flush=True)
    print(f"checkpoints={args.outdir}", flush=True)


if __name__ == "__main__":
    main()
