from __future__ import annotations

import torch


def frame_f1_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    valid = labels != -100
    preds = preds[valid]
    labels = labels[valid]

    out: dict[str, float] = {}
    class_f1 = []
    for class_id, name in [(0, 'english'), (1, 'hindi')]:
        tp = ((preds == class_id) & (labels == class_id)).sum().item()
        fp = ((preds == class_id) & (labels != class_id)).sum().item()
        fn = ((preds != class_id) & (labels == class_id)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[f'{name}_precision'] = precision
        out[f'{name}_recall'] = recall
        out[f'{name}_f1'] = f1
        class_f1.append(f1)
    out['macro_f1'] = sum(class_f1) / len(class_f1)
    out['accuracy'] = (preds == labels).float().mean().item() if labels.numel() else 0.0
    return out
