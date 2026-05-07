#!/usr/bin/env python3
"""
OPTUNA HYPERPARAMETER OPTIMIZATION (FIXED VERSION)
"""

import argparse, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent))
from models import create_model

# -----------------------------
# OBJECTIVE
# -----------------------------
def objective(trial, args):

    # 🔧 HYPERPARAMS
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    grad_clip = trial.suggest_float('grad_clip', 0.5, 1.5)

    if args.model == 'real_mamba':
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    # -----------------------------
    # MODEL
    # -----------------------------
    try:
        if args.model == 'real_mamba':
            model = create_model(args.model, input_size=40, d_model=d_model).to(device)
        else:
            model = create_model(args.model, input_size=40).to(device)
    except Exception as e:
        print(f"⚠️ Model error: {e}")
        return 0.0

    # -----------------------------
    # DATA (FIXED: 2 values, not 3!)
    # -----------------------------
    from dataset import create_dataloaders

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        seq_length=48,
        batch_size=batch_size
    )

    # -----------------------------
    # LOSS (imbalance-aware)
    # -----------------------------
    ys = []
    for _, _, y in train_loader:
        ys.append(y.numpy())
    ys = np.concatenate(ys)

    pos = np.mean(ys)
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    best_auc = 0
    patience = 3
    no_improve = 0

    for epoch in range(args.epochs):

        # ---- TRAIN ----
        model.train()
        for x, mask, y in train_loader:

            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(device.type == 'cuda'):
                out = model(x, mask)
                loss = criterion(out, y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # ---- VALIDATION ----
        model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for x, mask, y in val_loader:
                x = x.to(device)
                mask = mask.to(device)

                with torch.cuda.amp.autocast(device.type == 'cuda'):
                    logits = model(x, mask)
                    prob = torch.sigmoid(logits)

                preds.extend(prob.cpu().numpy())
                labels.extend(y.numpy())

        preds = np.array(preds)
        labels = np.array(labels)

        # ---- METRIC ----
        try:
            auc = roc_auc_score(labels, preds)
        except Exception:
            auc = 0.5

        # ---- REPORT TO OPTUNA ----
        trial.report(auc, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---- EARLY STOP ----
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_auc


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--n-trials', type=int, default=25)
    ap.add_argument('--timeout', type=int, default=7200)
    args = ap.parse_args()

    print("\n🚀 OPTUNA SEARCH")
    print("="*60)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5)
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    print("\n🏆 BEST RESULT")
    print("="*60)
    print(f"Best AUROC: {study.best_value:.4f}")

    print("\nBest params:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    Path("../logs").mkdir(exist_ok=True)

    with open(f"../logs/{args.model}_optuna.json", "w") as f:
        json.dump({
            "best_value": float(study.best_value),
            "best_params": study.best_params
        }, f, indent=2)

    print("\n✅ Saved results")


if __name__ == "__main__":
    main()
