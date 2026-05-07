#!/usr/bin/env python3
"""
Обучение Time-aware моделей
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models_timeaware import create_timeaware_model, count_parameters
from dataset_timeaware import create_dataloaders


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, mask, y in tqdm(loader, desc="Training", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x, mask)
        
        if torch.isnan(out).any():
            print(f"⚠️  NaN в выходе модели, пропускаем батч")
            continue
        
        loss = criterion(out.squeeze(-1), y)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  NaN/Inf loss, пропускаем батч")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(out.squeeze(-1)).detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5, posinf=0.999, neginf=0.001)
        all_preds.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy())
    
    return total_loss / max(len(loader), 1), all_preds, all_labels


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, mask, y in tqdm(loader, desc="Evaluating", leave=False):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        out = model(x, mask)
        
        if torch.isnan(out).any():
            print(f"⚠️  NaN в выходе модели, пропускаем батч")
            continue
        
        loss = criterion(out.squeeze(-1), y)
        total_loss += loss.item()
        
        probs = torch.sigmoid(out.squeeze(-1)).detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5, posinf=0.999, neginf=0.001)
        all_preds.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5
    
    return total_loss / max(len(loader), 1), auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['lstm_time', 'mamba_time'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-dir', type=str, default='../data/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
    parser.add_argument('--save-dir', type=str, default='../models')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TIME-AWARE: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    model = create_timeaware_model(args.model, input_size=42, d_model=128, hidden_size=128)
    model = model.to(device)
    print(f"Параметры: {count_parameters(model):,}\n")
    
    train_loader, val_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_auroc = 0
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, auroc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        print(f"\nEpoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  AUROC:      {auroc:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            Path(args.save_dir).mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"{args.save_dir}/{args.model}_best.pt")
            print(f"  💾 Сохранено! (AUROC: {auroc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"TIME-AWARE ЗАВЕРШЕНО")
    print(f"Лучший AUROC: {best_auroc:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
