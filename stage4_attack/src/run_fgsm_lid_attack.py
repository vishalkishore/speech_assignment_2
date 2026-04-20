from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio


PART_4_2_ROOT = Path(__file__).resolve().parents[1]  # stage4_attack/
PART_1_1_ROOT = Path(__file__).resolve().parents[2] / "stage1_lid"
if str(PART_1_1_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_1_1_ROOT))

from src.models.lid_bilstm import FrameLIDBiLSTM  # type: ignore  # noqa: E402


DEFAULT_CHECKPOINT = PART_1_1_ROOT / "checkpoints/lid_clip_adapt_clean_v2_best_clip.pt"
DEFAULT_SPANS = Path(__file__).resolve().parents[2] / "stage1_stt/reference/gemini_pro_spans_clean.csv"
DEFAULT_AUDIO_DIR = PART_4_2_ROOT / "audio"
DEFAULT_RESULTS_DIR = PART_4_2_ROOT / "results"

ID_TO_LANG = {0: "english", 1: "hindi"}
LANG_TO_ID = {"english": 0, "hindi": 1}


@dataclass
class CandidateSpan:
    utterance_id: str
    audio_path: Path
    span_start_sec: float
    span_end_sec: float
    language: str
    text: str

    @property
    def duration_sec(self) -> float:
        return self.span_end_sec - self.span_start_sec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted FGSM-style attack on the Part 1.1 LID model.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--spans-csv", type=Path, default=DEFAULT_SPANS)
    p.add_argument("--audio", type=Path, default=None, help="Optional direct audio path. If omitted, auto-pick a Hindi span.")
    p.add_argument("--segment-start-sec", type=float, default=None)
    p.add_argument("--duration-sec", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=400)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--smooth-window-frames", type=int, default=11)
    p.add_argument("--target-lang", choices=["english", "hindi"], default="english")
    p.add_argument("--snr-min-db", type=float, default=40.0)
    p.add_argument("--eps-min", type=float, default=5e-5)
    p.add_argument("--eps-max", type=float, default=8e-4)
    p.add_argument("--eps-steps", type=int, default=16)
    p.add_argument("--attack-steps", type=int, default=12, help="1 gives classic FGSM; >1 gives iterative FGSM/BIM.")
    p.add_argument("--save-prefix", default="attack_eval")
    return p.parse_args()


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def median_smooth_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = torch.cat([x[:1].repeat(pad), x, x[-1:].repeat(pad)], dim=0)
    cols = [x_pad[i : i + x.numel()] for i in range(window)]
    return torch.stack(cols, dim=1).median(dim=1).values.to(dtype=x.dtype)


class LIDAttackModel:
    def __init__(self, args: argparse.Namespace) -> None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        self.device = torch.device(args.device)
        self.model = FrameLIDBiLSTM()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = args.sample_rate
        self.hop_length = args.hop_length
        self.smooth_window_frames = args.smooth_window_frames
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            center=False,
            power=2.0,
        ).to(self.device)

    def logits(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        feats = self.mel(wav)
        feats = torch.log(feats.clamp_min(1e-5)).transpose(1, 2)
        return self.model(feats)

    def predict(self, wav: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            logits = self.logits(wav.to(self.device)).squeeze(0).cpu()
        raw = logits.argmax(dim=-1).to(dtype=torch.int64)
        smooth = median_smooth_1d(raw, self.smooth_window_frames)
        english_frames = int((smooth == 0).sum().item())
        hindi_frames = int((smooth == 1).sum().item())
        total = int(smooth.numel()) or 1
        mean_logits = logits.mean(dim=0)
        mean_probs = logits.softmax(dim=-1).mean(dim=0)
        pred_id = 0 if english_frames >= hindi_frames else 1
        return {
            "pred_id": pred_id,
            "pred_lang": ID_TO_LANG[pred_id],
            "english_ratio": english_frames / total,
            "hindi_ratio": hindi_frames / total,
            "num_frames": total,
            "mean_logits": mean_logits.tolist(),
            "mean_probs": mean_probs.tolist(),
        }


def candidate_spans_from_csv(path: Path, min_duration_sec: float, target_language: str) -> list[CandidateSpan]:
    rows: list[CandidateSpan] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"].strip().lower() != target_language:
                continue
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if (end_sec - start_sec) < min_duration_sec:
                continue
            rows.append(
                CandidateSpan(
                    utterance_id=row["utterance_id"],
                    audio_path=Path(row["audio_path"]),
                    span_start_sec=start_sec,
                    span_end_sec=end_sec,
                    language=row["language"].strip().lower(),
                    text=row["text"].strip(),
                )
            )
    rows.sort(key=lambda x: x.duration_sec, reverse=True)
    return rows


def choose_attack_segment(
    model: LIDAttackModel,
    spans_csv: Path,
    duration_sec: float,
) -> tuple[CandidateSpan, float, torch.Tensor, dict[str, Any]]:
    candidates = candidate_spans_from_csv(spans_csv, duration_sec, target_language="hindi")
    for cand in candidates:
        wav = load_audio_mono(cand.audio_path, model.sample_rate)
        max_start = max(cand.span_start_sec, cand.span_end_sec - duration_sec)
        start_sec = cand.span_start_sec + max(0.0, (cand.duration_sec - duration_sec) / 2.0)
        start_sec = min(start_sec, max_start)
        start_idx = int(round(start_sec * model.sample_rate))
        end_idx = start_idx + int(round(duration_sec * model.sample_rate))
        segment = wav[start_idx:end_idx]
        if segment.numel() != int(round(duration_sec * model.sample_rate)):
            continue
        pred = model.predict(segment)
        if pred["pred_lang"] == "hindi":
            return cand, start_sec, segment, pred
    raise RuntimeError("No clean Hindi 5-second segment stayed Hindi under the current LID model.")


def snr_db(clean: torch.Tensor, adv: torch.Tensor) -> float:
    noise = adv - clean
    signal_power = clean.pow(2).mean().item()
    noise_power = noise.pow(2).mean().item()
    if noise_power <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))


def run_targeted_attack(
    model: LIDAttackModel,
    clean_wav: torch.Tensor,
    target_id: int,
    epsilon: float,
    attack_steps: int,
) -> torch.Tensor:
    clean = clean_wav.to(model.device)
    adv = clean.clone()
    alpha = epsilon if attack_steps <= 1 else epsilon / attack_steps
    target = None
    for _ in range(max(1, attack_steps)):
        adv = adv.detach().clone().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            logits = model.logits(adv).squeeze(0)
            target = torch.full((logits.size(0),), target_id, dtype=torch.long, device=model.device)
            loss = F.cross_entropy(logits, target)
            grad = torch.autograd.grad(loss, adv)[0]
        adv = adv - alpha * grad.sign()
        delta = torch.clamp(adv - clean, min=-epsilon, max=epsilon)
        adv = torch.clamp(clean + delta, min=-1.0, max=1.0)
    return adv.detach().cpu()


def epsilon_grid(eps_min: float, eps_max: float, num_steps: int) -> list[float]:
    if num_steps <= 1:
        return [eps_max]
    return torch.linspace(eps_min, eps_max, steps=num_steps).tolist()


def main() -> None:
    args = parse_args()
    model = LIDAttackModel(args)

    if args.audio is None:
        cand, start_sec, clean_segment, clean_pred = choose_attack_segment(model, args.spans_csv, args.duration_sec)
        selection = {
            "selection_mode": "auto_from_clean_hindi_spans",
            "utterance_id": cand.utterance_id,
            "audio_path": str(cand.audio_path),
            "reference_span_start_sec": cand.span_start_sec,
            "reference_span_end_sec": cand.span_end_sec,
            "attack_segment_start_sec": start_sec,
            "attack_segment_end_sec": start_sec + args.duration_sec,
            "reference_text": cand.text,
        }
    else:
        if args.segment_start_sec is None:
            raise ValueError("--segment-start-sec is required when --audio is provided.")
        wav = load_audio_mono(args.audio, model.sample_rate)
        start_idx = int(round(args.segment_start_sec * model.sample_rate))
        end_idx = start_idx + int(round(args.duration_sec * model.sample_rate))
        clean_segment = wav[start_idx:end_idx]
        if clean_segment.numel() != int(round(args.duration_sec * model.sample_rate)):
            raise ValueError("Requested segment extends beyond the audio file.")
        clean_pred = model.predict(clean_segment)
        selection = {
            "selection_mode": "manual",
            "audio_path": str(args.audio),
            "attack_segment_start_sec": args.segment_start_sec,
            "attack_segment_end_sec": args.segment_start_sec + args.duration_sec,
        }

    clean_lang = clean_pred["pred_lang"]
    target_id = LANG_TO_ID[args.target_lang]
    if clean_lang == args.target_lang:
        raise RuntimeError(f"Clean segment is already predicted as {args.target_lang}; choose a Hindi segment instead.")

    best: dict[str, Any] | None = None
    best_success: dict[str, Any] | None = None
    for epsilon in epsilon_grid(args.eps_min, args.eps_max, args.eps_steps):
        adv = run_targeted_attack(model, clean_segment, target_id=target_id, epsilon=epsilon, attack_steps=args.attack_steps)
        adv_pred = model.predict(adv)
        snr = snr_db(clean_segment, adv)
        record = {
            "epsilon": float(epsilon),
            "snr_db": float(snr),
            "prediction": adv_pred,
            "flip_success": adv_pred["pred_lang"] == args.target_lang,
        }
        if best is None or adv_pred["mean_probs"][target_id] > best["record"]["prediction"]["mean_probs"][target_id]:
            best = {"record": record, "adv": adv}
        if record["flip_success"] and snr > args.snr_min_db:
            best_success = {"record": record, "adv": adv}
            break

    DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_audio_path = DEFAULT_AUDIO_DIR / f"{args.save_prefix}_clean.wav"
    torchaudio.save(str(clean_audio_path), clean_segment.unsqueeze(0), model.sample_rate)

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "target_lang": args.target_lang,
        "snr_min_db": args.snr_min_db,
        "attack_steps": args.attack_steps,
        "selection": selection,
        "clean_prediction": clean_pred,
        "clean_audio_path": str(clean_audio_path),
        "epsilon_grid": epsilon_grid(args.eps_min, args.eps_max, args.eps_steps),
        "success": best_success is not None,
    }

    if best_success is not None:
        adv_audio_path = DEFAULT_AUDIO_DIR / f"{args.save_prefix}_adv.wav"
        torchaudio.save(str(adv_audio_path), best_success["adv"].unsqueeze(0), model.sample_rate)
        result["best_attack"] = best_success["record"]
        result["adv_audio_path"] = str(adv_audio_path)
    elif best is not None:
        adv_audio_path = DEFAULT_AUDIO_DIR / f"{args.save_prefix}_best_effort.wav"
        torchaudio.save(str(adv_audio_path), best["adv"].unsqueeze(0), model.sample_rate)
        result["best_attack"] = best["record"]
        result["adv_audio_path"] = str(adv_audio_path)
        result["note"] = "No attack met the SNR threshold and forced a flip; saved the strongest best-effort attack."
    else:
        raise RuntimeError("Attack loop produced no candidate outputs.")

    result_path = DEFAULT_RESULTS_DIR / f"{args.save_prefix}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
