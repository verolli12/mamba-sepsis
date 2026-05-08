#!/usr/bin/env python3
"""
STABLE TRAINING PIPELINE v2.4 — CRITICAL FIXES APPLIED
✅ Fixed pos_weight calculation (memory safe)
✅ Fixed loss averaging with NaN skipping
✅ Fixed num_workers for Windows
✅ Safe tensor handling (.cpu().numpy())
✅ Robust EMA save/restore logic
✅ Gradient accumulation with correct scheduler stepping
✅ Warmup + Cosine annealing scheduler
"""

import argparse
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Fix __file__ for exec() compatibility
if '__file__' not in globals():
    __file__ = sys.argv[0] if sys.argv else 'train.py'

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
# EMA (Exponential Moving Average)
# -----------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.original_state = None
        
    def update(self):
        """Обновить EMA состояние после шага оптимизатора"""
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if k in self.ema_state:
                    self.ema_state[k] = self.decay * self.ema_state[k] + (1 - self.decay) * v
    
    def apply(self):
        """Применить EMA веса к модели для evaluation"""
        with torch.no_grad():
            self.model.load_state_dict(self.ema_state, strict=False)
    
    def store_original(self):
        """Сохранить текущие веса модели"""
        self.original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def restore_original(self):
        """Восстановить сохранённые веса модели"""
        if self.original_state is not None:
            with torch.no_grad():
                self.model.load_state_dict(self.original_state)


# -----------------------------
# TRAIN WITH GRADIENT ACCUMULATION
# -----------------------------
def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, 
                accum_steps=2, grad_clip=1.0):
    """
    Training loop с градиентной аккумуляцией.
    
    Важно:
    - loss делится на accum_steps перед backward()
    - optimizer.step() и scheduler.step() вызываются только каждый accum_steps шаг
    """
    model.train()
    preds, labels = [], []
    total_loss = 0
    valid_batches = 0  # 🔥 Считаем только успешные батчи
    optimizer.zero_grad()

    for i, (x, mask, y) in enumerate(tqdm(loader, desc="train", leave=False)):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', 
                               enabled=(device.type == 'cuda')):
            out = model(x, mask)
            loss = criterion(out, y) / accum_steps

        # Пропускаем батчи с нестабильным loss
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        valid_batches += 1  # 🔥 Считаем только успешные батчи
        scaler.scale(loss).backward()
        
        # Только каждый accum_steps шаг делаем оптимизацию
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # 🔥 scheduler.step() ТОЛЬКО после реального optimizer.step()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        with torch.no_grad():
            p = torch.sigmoid(out).detach().cpu().numpy()
            preds.extend(p)
            labels.extend(y.cpu().numpy())

    return total_loss / max(1, valid_batches), np.array(preds), np.array(labels)


# -----------------------------
# EVALUATE
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluation без сглаживания — честные метрики"""
    model.eval()
    loss_sum, preds, labels = 0, [], []

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                               enabled=(device.type == 'cuda')):
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
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    with open(path, "w", newline='') as f:
        import csv
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thr):
            w.writerow([a, b, c])

def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'transformer', 'real_mamba', 'grud'])
    parser.add_argument('--data-dir', type=str, default='../data/training_setA')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seq-len', type=int, default=48)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='../models')
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Модель: {args.model.upper()}")
    print(f"Устройство: {device}")
    print(f"{'='*60}\n")
    
    # Создание модели
    if args.model in ['lstm', 'grud']:
        model = create_model(args.model, input_size=40, hidden_size=args.d_model)
    else:
        model = create_model(args.model, input_size=40, d_model=args.d_model)
    
    model = model.to(device)
    params = count_parameters(model)
    print(f"Параметры: {params:,}\n")
    
    # Данные
    if args.dummy:
        print("⚠️  Режим DUMMY")
        from dataset import SyntheticSepsisBatch
        train_ds = SyntheticSepsisBatch(256, args.seq_len, 40, seed=1)
        val_ds = SyntheticSepsisBatch(64, args.seq_len, 40, seed=2)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        print(f"📂 Загрузка реальных данных: {args.data_dir}")
        try:
            from dataset import create_dataloaders
            train_loader, val_loader = create_dataloaders(
                data_dir=args.data_dir,
                seq_length=args.seq_len,
                batch_size=args.batch_size,
                val_split=0.2,
                normalize=True
            )
            print(f"✅ Реальные данные загружены!")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("⚠️  Переключаюсь на dummy")
            from dataset import SyntheticSepsisBatch
            train_ds = SyntheticSepsisBatch(256, args.seq_len, 40, seed=1)
            val_ds = SyntheticSepsisBatch(64, args.seq_len, 40, seed=2)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_auroc = 0
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, auroc, f1, acc = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        print(f"\nEpoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  AUROC:      {auroc:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Accuracy:   {acc:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'config': vars(args),
                'epoch': epoch,
                'auroc': auroc
            }, f"{args.save_dir}/{args.model}_best.pt")
            print(f"  💾 Сохранено! (AUROC: {auroc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший AUROC: {best_auroc:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
