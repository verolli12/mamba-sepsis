#!/usr/bin/env python3
"""
STABLE TRAINING PIPELINE v2.0
✅ Исправлены все баги
✅ Градиентная аккумуляция работает корректно
✅ Совместим с models.py
✅ FutureWarnings исправлены
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from models import create_model, count_parameters

# -----------------------------
# WARMUP + COSINE SCHEDULER
# -----------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_num < self.warmup_steps:
                lr = base_lr * self.step_num / max(1, self.warmup_steps)
            else:
                progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
            pg['lr'] = lr

# -----------------------------
# TRAIN EPOCH (FIXED)
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, 
                accum_steps=1, grad_clip=1.0):
    """
    Training с опциональной градиентной аккумуляцией.
    
    Важно:
    - Если accum_steps > 1, loss делится на accum_steps ПЕРЕД backward()
    - optimizer.step() вызывается только каждый accum_steps шаг
    - scheduler.step() вызывается только после реального обновления весов
    """
    model.train()
    preds, labels = [], []
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (x, mask, y) in enumerate(tqdm(loader, desc="train", leave=False)):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        # 🔥 FIX: Используем новый API для autocast
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            out = model(x, mask)
            loss = criterion(out, y)
            
            # 🔥 FIX: Нормализуем loss ТОЛЬКО если используем аккумуляцию
            if accum_steps > 1:
                loss = loss / accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Skipping batch {i}: loss={loss.item()}")
            continue

        # Backward pass
        scaler.scale(loss).backward()
        
        # 🔥 FIX: Только каждый accum_steps шаг делаем оптимизацию
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # 🔥 FIX: Scheduler шаг ТОЛЬКО после реального обновления весов
            scheduler.step()

        total_loss += loss.item() * (accum_steps if accum_steps > 1 else 1)
        
        # Collect predictions for metrics
        with torch.no_grad():
            p = torch.sigmoid(out).detach().cpu().numpy()
            preds.extend(p)
            labels.extend(y.cpu().numpy())

    return total_loss / max(1, len(loader)), np.array(preds), np.array(labels)

# -----------------------------
# EVALUATE (FIXED)
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation без багов"""
    model.eval()
    loss_sum, preds, labels = 0, [], []

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            out = torch.sigmoid(logits)
        
        preds.extend(out.cpu().numpy())
        labels.extend(y.cpu().numpy())

    preds, labels = np.array(preds), np.array(labels)
    
    if len(np.unique(labels)) < 2:
        return loss_sum / max(1, len(loader)), 0.5, 0.0, 0.5, preds, labels

    return (
        loss_sum / len(loader),
        roc_auc_score(labels, preds),
        f1_score(labels, (preds > 0.5).astype(int), zero_division=0),
        accuracy_score(labels, (preds > 0.5).astype(int)),
        preds, labels
    )

# -----------------------------
# SAVE UTILS
# -----------------------------
def save_roc(y_true, y_prob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    with open(path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thr):
            w.writerow([a, b, c])

def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Stable Training Pipeline")
    ap.add_argument('--model', required=True, choices=['lstm', 'transformer', 'real_mamba', 'grud'])
    ap.add_argument('--data-dir', type=str, default='../data/training_setA')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seq-len', type=int, default=48)
    ap.add_argument('--dummy', action='store_true', help='Use synthetic data for testing')
    ap.add_argument('--save-dir', type=str, default='../models')
    ap.add_argument('--log-dir', type=str, default='../logs')
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation steps (1 = disabled)')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pos-weight', type=float, default=None, help='Manual positive class weight')
    args = ap.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 🔥 FIX: MODEL CREATION - без d_model чтобы совместимо с models.py
    print(f"Creating model: {args.model}")
    model = create_model(args.model, input_size=40).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # DATA LOADING
    print(f"Loading data from: {args.data_dir}")
    if args.dummy:
        from dataset import SyntheticSepsisBatch
        train_ds = SyntheticSepsisBatch(512, args.seq_len, 40)
        val_ds = SyntheticSepsisBatch(128, args.seq_len, 40)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    else:
        from dataset import create_dataloaders
        # 🔥 FIX: create_dataloaders возвращает 2 значения, НЕ 3!
        train_loader, val_loader = create_dataloaders(args.data_dir, args.seq_len, args.batch_size)

    # CLASS IMBALANCE: Auto pos_weight
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight], device=device)
    else:
        print("Computing class imbalance weight...")
        all_y = []
        for _, _, y in train_loader:
            all_y.extend(y.numpy())
        pos = np.mean(all_y)
        pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
        print(f"  Positive class: {pos*100:.2f}% → pos_weight = {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # SCHEDULER
    # 🔥 FIX: total_steps учитывает accum_steps
    effective_batches = len(train_loader) // args.accum_steps if args.accum_steps > 1 else len(train_loader)
    total_steps = args.epochs * effective_batches
    scheduler = WarmupCosine(optimizer, args.warmup, total_steps)

    # 🔥 FIX: GradScaler с новым API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # TRAINING LOOP
    best_auc, best_model_state = 0, None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}

    print(f"\n🚀 Starting training: {args.epochs} epochs")
    print(f"   Batches/epoch: {len(train_loader)}")
    print(f"   Accum steps: {args.accum_steps}")
    print(f"   Effective LR: {args.lr}{' (scaled)' if args.accum_steps > 1 else ''}")
    print("=" * 70)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # TRAIN
        train_loss, tr_preds, tr_labels = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device,
            accum_steps=args.accum_steps, grad_clip=args.grad_clip
        )
        
        # EVALUATE
        val_loss, auc, f1, acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # LOG
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_auc'].append(float(auc))
        history['val_f1'].append(float(f1))
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} | val_auc={auc:.4f} | val_f1={f1:.4f} | "
              f"time={epoch_time:.1f}s")
        
        # EARLY STOPPING + SAVE BEST
        if auc > best_auc:
            best_auc = auc
            best_f1 = f1
            best_epoch = epoch + 1
            no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': auc,
                'val_f1': f1,
                'args': vars(args)
            }, Path(args.save_dir) / f"{args.model}_best.pt")
            print(f"  ✅ New best! Saved to {args.save_dir}/{args.model}_best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print(f"🏆 Best AUROC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"🎯 Best F1: {best_f1:.4f}")
    
    # LOAD BEST MODEL FOR FINAL EVAL
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model for final evaluation")
    
    # FINAL EVALUATION
    _, _, _, _, vp, vy = evaluate(model, val_loader, criterion, device)
    
    # THRESHOLD OPTIMIZATION (by F1)
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(vy, vp)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_threshold = thr[np.argmax(f1_scores)] if len(thr) > 0 else 0.5
    print(f"📊 Optimal threshold: {best_threshold:.3f}")
    
    # SAVE OUTPUTS
    save_roc(vy, vp, Path(args.log_dir) / f"{args.model}_roc.csv")
    
    metrics = {
        'model': args.model,
        'best_auc': float(best_auc),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'best_epoch': int(best_epoch),
        'epochs_trained': epoch + 1,
        'config': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'accum_steps': args.accum_steps,
            'grad_clip': args.grad_clip,
            'warmup': args.warmup,
            'patience': args.patience,
            'seed': args.seed
        },
        'history': history
    }
    save_metrics(metrics, Path(args.log_dir) / f"{args.model}_metrics.json")
    print(f"✅ Metrics saved: {args.log_dir}/{args.model}_metrics.json")
    
    # TRAINING PLOT
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=1)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=1)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metrics plot
        axes[1].plot(history['val_auc'], label='Val AUROC', color='#2ecc71', linewidth=2)
        axes[1].plot(history['val_f1'], label='Val F1', color='#e74c3c', linewidth=2)
        axes[1].axhline(y=best_auc, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Best AUROC={best_auc:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(args.log_dir) / f"{args.model}_training.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Plot saved: {plot_path}")
    except Exception as e:
        print(f"⚠️ Could not save training plot: {e}")
    
    print("\n🎉 TRAINING COMPLETE!")
    return best_auc

if __name__ == "__main__":
    main()
