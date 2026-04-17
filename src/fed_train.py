#!/usr/bin/env python3
"""
Федеративное обучение для Mamba/LSTM/Transformer
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model, count_parameters


class PhysioNetDataset(Dataset):
    def __init__(self, file_list, seq_length=48):
        self.files = file_list
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        
        print(f"  📂 Загрузка {len(file_list)} файлов...")
        for f in tqdm(file_list[:100], desc="    Loading"):
            try:
                df = pd.read_csv(f, sep='|')
                if len(df) > seq_length:
                    df = df.iloc[-seq_length:]
                else:
                    df = df.iloc[-seq_length:].reindex(range(seq_length), method='bfill')
                
                features = [c for c in df.columns if c != 'SepsisLabel']
                x = df[features].fillna(0).values.astype(np.float32)
                x = np.nan_to_num(x, nan=0, posinf=1e6, neginf=-1e6)
                
                y = float(df['SepsisLabel'].fillna(0).iloc[-1])
                
                self.data.append(torch.tensor(x, dtype=torch.float32))
                self.labels.append(torch.tensor(y, dtype=torch.float32))
            except:
                continue
        
        print(f"  ✅ Загружено {len(self.data)} образцов")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        mask = torch.ones_like(x)
        y = self.labels[idx]
        return x, mask, y


def load_client_files(client_file):
    with open(client_file, 'r') as f:
        return [Path(line.strip()) for line in f if line.strip()]


def train_local_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x, mask)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(len(loader), 1)


def federated_average(model_states, weights=None):
    if weights is None:
        weights = [1.0 / len(model_states)] * len(model_states)
    
    averaged = {}
    for key in model_states[0].keys():
        averaged[key] = sum(w * ms[key] for w, ms in zip(weights, model_states))
    
    return averaged


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            out = model(x, mask)
            loss = criterion(out, y)
            total_loss += loss.item()
    
    return total_loss / max(len(loader), 1)


def create_model_safe(model_name, input_size=40, d_model=128, hidden_size=128):
    """Безопасное создание модели"""
    if model_name in ['lstm', 'grud']:
        return create_model(model_name, input_size=input_size, hidden_size=hidden_size)
    else:
        return create_model(model_name, input_size=input_size, d_model=d_model)


def train_federated(model_name, client_files, rounds=5, local_epochs=3, 
                    lr=0.0001, batch_size=16, val_files=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ: {model_name.upper()}")
    print(f"Клиентов: {len(client_files)}, Раундов: {rounds}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Глобальная модель
    global_model = create_model_safe(model_name, input_size=40, d_model=128, hidden_size=128)
    global_model = global_model.to(device)
    
    print(f"Параметры: {count_parameters(global_model):,}\n")
    
    criterion = nn.BCEWithLogitsLoss()
    
    if val_files:
        val_dataset = PhysioNetDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    best_loss = float('inf')
    
    for round_num in range(1, rounds + 1):
        print(f"\n🔄 Round {round_num}/{rounds}")
        
        client_states = []
        client_sizes = []
        
        for client_idx, client_file in enumerate(client_files):
            print(f"\n  👤 Client {client_idx}:")
            
            files = load_client_files(client_file)
            if len(files) == 0:
                print(f"    ⚠️  Нет файлов, пропускаем")
                continue
            
            dataset = PhysioNetDataset(files)
            if len(dataset) == 0:
                print(f"    ⚠️  Пустой датасет, пропускаем")
                continue
            
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Локальная модель
            local_model = create_model_safe(model_name, input_size=40, d_model=128, hidden_size=128)
            local_model.load_state_dict(global_model.state_dict())
            local_model = local_model.to(device)
            
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
            
            for epoch in range(local_epochs):
                loss = train_local_epoch(local_model, loader, criterion, optimizer, device)
            
            client_states.append({k: v.cpu().clone() for k, v in local_model.state_dict().items()})
            client_sizes.append(len(dataset))
            
            print(f"    ✅ Обучено, loss={loss:.4f}")
        
        if len(client_states) == 0:
            print("  ⚠️  Ни один клиент не обучился!")
            continue
        
        weights = [s / sum(client_sizes) for s in client_sizes]
        averaged_params = federated_average(client_states, weights)
        
        global_model.load_state_dict(averaged_params)
        
        if val_loader:
            val_loss = evaluate(global_model, val_loader, criterion, device)
            print(f"\n  📊 Global Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                Path("../models").mkdir(exist_ok=True)
                torch.save(global_model.state_dict(), 
                          f"../models/{model_name}_fed_best.pt")
                print(f"  💾 Сохранено! (Val Loss: {val_loss:.4f})")
        
        print(f"  ✅ Round {round_num} завершён")
    
    print(f"\n{'='*60}")
    print(f"ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучший Val Loss: {best_loss:.4f}")
    print(f"{'='*60}\n")
    
    return global_model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='real_mamba',
                       choices=['lstm', 'transformer', 'real_mamba'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--local-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--split-dir', type=str, default='../data/fed_splits')
    args = parser.parse_args()
    
    split_dir = Path(args.split_dir)
    client_files = [split_dir / f"client_{i}.txt" for i in range(5)]
    client_files = [f for f in client_files if f.exists()]
    
    print(f"📁 Найдено клиентов: {len(client_files)}")
    
    model = train_federated(
        model_name=args.model,
        client_files=client_files,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    Path("../models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"../models/{args.model}_fed_final.pt")
    print(f"💾 Модель сохранена: ../models/{args.model}_fed_final.pt")
