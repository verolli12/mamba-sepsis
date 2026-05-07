#!/usr/bin/env python3
"""Обучение всех моделей (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model, count_parameters


class SyntheticSepsisBatch(torch.utils.data.Dataset):
    def __init__(self, n, seq_len, input_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, seq_len, input_size, generator=g)
        self.mask = torch.ones_like(self.x)
        self.y = torch.randint(0, 2, (n,), generator=g).float()
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        return self.x[i], self.mask[i], self.y[i]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, mask, y in tqdm(loader, desc="Training", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x, mask)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # ✅ ИСПРАВЛЕНО: добавлено .detach()
        all_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    return total_loss / len(loader), all_preds, all_labels


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, mask, y in tqdm(loader, desc="Evaluating", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        out = model(x, mask)
        loss = criterion(out, y)
        
        total_loss += loss.item()
        # ✅ ИСПРАВЛЕНО: добавлено .detach()
        all_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (torch.tensor(all_preds) > 0.5).numpy())
    acc = accuracy_score(all_labels, (torch.tensor(all_preds) > 0.5).numpy())
    
    return total_loss / len(loader), auroc, f1, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'transformer', 'real_mamba', 'grud'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seq-len', type=int, default=48)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--dummy', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Модель: {args.model.upper()}")
    print(f"Устройство: {device}")
    print(f"{'='*60}\n")
    
    # ✅ ИСПРАВЛЕНО: правильные параметры для каждой модели
    if args.model in ['lstm', 'grud']:
        model = create_model(args.model, input_size=40, hidden_size=args.d_model)
    else:
        model = create_model(args.model, input_size=40, d_model=args.d_model)
    
    model = model.to(device)
    params = count_parameters(model)
    print(f"Параметры: {params:,}\n")
    
    # Данные
    if args.dummy:
        print("⚠️  Режим DUMMY (синтетические данные)")
        train_ds = SyntheticSepsisBatch(256, args.seq_len, 40, seed=1)
        val_ds = SyntheticSepsisBatch(64, args.seq_len, 40, seed=2)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        print("❌ Real data not configured. Use --dummy for testing.")
        sys.exit(1)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_auroc = 0
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, auroc, f1, acc = evaluate(
            model, val_loader, criterion, device
        )
        
        elapsed = time.time() - start
        
        print(f"\nEpoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  AUROC:      {auroc:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  Accuracy:   {acc:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            print(f"  💾 Новый рекорд! (AUROC: {auroc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший AUROC: {best_auroc:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
