#!/usr/bin/env python3
"""
Обучение всех моделей для курсовой работы
ИСПРАВЛЕННАЯ ВЕРСИЯ - с защитой от NaN
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model, count_parameters


def train_epoch(model, loader, criterion, optimizer, device):
    """Обучение на одной эпохе с защитой от NaN"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (x, mask, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        try:
            out = model(x, mask)
            
            # Проверка на NaN
            if torch.isnan(out).any():
                print("⚠️ NaN in output, skipping batch")
                continue
            
            loss = criterion(out, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ NaN in loss, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # 🔍 DEBUG: Проверка градиентов
            if batch_idx == 0 and epoch == 1:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"🔍 Gradient norm: {total_norm:.6f}")
                if total_norm < 1e-7:
                    print("⚠️  ГРАДИЕНТЫ ИСЧЕЗАЮТ! Модель не обучается!")
            
            total_loss += loss.item()
            
            # Безопасные предсказания
            with torch.no_grad():
                probs = torch.sigmoid(out).cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.5, posinf=0.999, neginf=0.001)
                probs = np.clip(probs, 0.001, 0.999)
            
            all_preds.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy())
            
        except Exception as e:
            print(f"⚠️ Error in batch: {e}")
            continue
    
    if len(all_preds) == 0:
        return float('inf'), [], []
    
    return total_loss / max(len(loader), 1), all_preds, all_labels


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Валидация с защитой от NaN"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, mask, y in tqdm(loader, desc="Evaluating", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        try:
            out = model(x, mask)
            loss = criterion(out, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(out).detach().cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5, posinf=0.999, neginf=0.001)
            probs = np.clip(probs, 0.001, 0.999)
            
            all_preds.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy())
            
        except Exception as e:
            print(f"⚠️ Error in evaluate: {e}")
            continue
    
    # Конвертация
    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    
    # Очистка
    valid = np.isfinite(all_preds) & np.isfinite(all_labels)
    all_preds = all_preds[valid]
    all_labels = all_labels[valid]
    
    if len(all_preds) == 0:
        return total_loss / max(len(loader), 1), 0.5, 0.0, 0.5
    
    # Безопасные метрики
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except Exception as e:
        print(f"⚠️ AUROC error: {e}")
        auroc = 0.5
    
    try:
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), zero_division=0)
    except Exception as e:
        print(f"⚠️ F1 error: {e}")
        f1 = 0.0
    
    try:
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    except Exception as e:
        print(f"⚠️ Accuracy error: {e}")
        acc = 0.5
    
    return total_loss / max(len(loader), 1), auroc, f1, acc


def main():
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
            model, train_loader, criterion, optimizer, device
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
