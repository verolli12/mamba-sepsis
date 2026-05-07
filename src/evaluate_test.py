#!/usr/bin/env python3
"""Final test-only evaluation for saved checkpoints.

Protocol:
1) create deterministic split (same seed as training)
2) load *_best.pt
3) evaluate once on test split
4) save metrics + ROC
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, auc, roc_curve

from dataset import create_dataloaders
from models import create_model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        preds.extend(torch.sigmoid(logits).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return loss_sum / max(1, len(loader)), np.array(preds), np.array(labels)


def save_roc(y_true, y_prob, path):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thr):
            w.writerow([a, b, c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "transformer", "real_mamba", "grud"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="../data/training_setA")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--test-split", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=None, help="Fixed threshold. If omitted, loaded from --threshold-json or defaults to 0.5")
    ap.add_argument("--threshold-json", type=str, default=None, help="Path to JSON with {'best_threshold': float} obtained on validation split")
    ap.add_argument("--manifest", type=str, default="../logs/split_manifest_test_eval.json")
    ap.add_argument("--out", type=str, default="../logs/test_metrics.json")
    ap.add_argument("--roc-out", type=str, default="../logs/test_roc.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_dataloaders(
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

    criterion = nn.BCEWithLogitsLoss()
    test_loss, y_prob, y_true = evaluate(model, test_loader, criterion, device)

    if len(np.unique(y_true)) < 2:
        test_auc = 0.5
        pr_auc = 0.0
    else:
        test_auc = roc_auc_score(y_true, y_prob)
        p, r, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r, p)

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.threshold_json is not None:
        with open(args.threshold_json, "r", encoding="utf-8") as f:
            threshold = float(json.load(f)["best_threshold"])
    else:
        threshold = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "threshold": threshold,
        "test_loss": float(test_loss),
        "test_auc": float(test_auc),
        "test_pr_auc": float(pr_auc),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "positive_rate_test": float(np.mean(y_true)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_roc(y_true, y_prob, args.roc_out)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
