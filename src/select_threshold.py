#!/usr/bin/env python3
"""Select decision threshold on validation split only.

This script MUST be run after training and before final test evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve

from dataset import create_dataloaders
from models import create_model


@torch.no_grad()
def collect_val_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        logits = model(x, mask)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(probs), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer", "real_mamba", "grud"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="../data/training_setA")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="../logs/threshold.json")
    ap.add_argument("--manifest", type=str, default="../logs/split_manifest_threshold.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        seq_length=args.seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        normalize=True,
        seed=args.seed,
        include_test=True,
        split_manifest_path=args.manifest,
    )

    model = create_model(args.model, input_size=40).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    y_prob, y_true = collect_val_probs(model, val_loader, device)

    if len(np.unique(y_true)) < 2:
        best_threshold, best_f1 = 0.5, 0.0
    else:
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        if len(thr) == 0:
            best_threshold, best_f1 = 0.5, float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0))
        else:
            idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > len(thr) else int(np.argmax(f1_scores))
            idx = min(idx, len(thr) - 1)
            best_threshold = float(thr[idx])
            best_f1 = float(f1_score(y_true, (y_prob >= best_threshold).astype(int), zero_division=0))

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "selection_split": "validation",
        "best_threshold": best_threshold,
        "best_f1_val": best_f1,
        "n_val": int(len(y_true)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
