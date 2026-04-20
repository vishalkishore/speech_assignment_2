from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from manifest_loader import load_rows_by_split
from lfcc_cm import LFCCCMNet, LFCCFrontend, SpoofWindowDataset, eer_from_scores, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LFCC anti-spoofing CM.")
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def evaluate(model, loader, device: str) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    eer = eer_from_scores(np.asarray(all_labels), np.asarray(all_scores))
    return {"accuracy": correct / max(total, 1), **eer}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    splits = load_rows_by_split(args.manifest_csv)
    frontend = LFCCFrontend()
    train_ds = SpoofWindowDataset(splits["train"], frontend)
    valid_ds = SpoofWindowDataset(splits["valid"], frontend)
    test_ds = SpoofWindowDataset(splits["test"], frontend)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LFCCCMNet().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss()

    history = []
    best_eer = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(y.numel())
            n += int(y.numel())
        valid = evaluate(model, valid_loader, args.device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n, 1),
            "valid_accuracy": valid["accuracy"],
            "valid_eer": valid["eer"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_eer is None or valid["eer"] < best_eer:
            best_eer = valid["eer"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "valid_metrics": valid,
                    "history": history,
                },
                args.outdir / "final_lfcc_cm.pt",
            )

    ckpt = torch.load(args.outdir / "final_lfcc_cm.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, args.device)

    (args.outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.outdir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps({"test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
