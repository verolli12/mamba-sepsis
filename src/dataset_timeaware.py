#!/usr/bin/env python3
"""
Time-aware датасет с временными метками
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PhysioNetTimeAware(Dataset):
    def __init__(self, data_dir, seq_length=48, normalize=True, max_files=5000):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.files = list(self.data_dir.glob("*.psv"))[:max_files]
        
        print(f"📂 Найдено файлов: {len(self.files)}")
        
        self.normalize = normalize
        self.mean = np.zeros(40, dtype=np.float32)
        self.std = np.ones(40, dtype=np.float32)
        
        if normalize:
            self._compute_stats()
    
    def _compute_stats(self):
        print("⏳ Вычисление статистики...")
        all_values = []
        for f in self.files[:100]:
            try:
                df = pd.read_csv(f, sep='|')
                features = [c for c in df.columns if c != 'SepsisLabel']
                values = df[features].values.astype(np.float32)
                values = np.nan_to_num(values, nan=0, posinf=1e6, neginf=-1e6)
                all_values.append(values)
            except:
                continue
        
        if all_values:
            all_values = np.vstack(all_values)
            self.mean = np.nanmean(all_values, axis=0).astype(np.float32)
            self.std = np.nanstd(all_values, axis=0).astype(np.float32)
            self.std = np.clip(self.std, 1e-6, 1e6)
            print(f"  ✅ Mean: {self.mean.shape}, Std: {self.std.shape}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            df = pd.read_csv(self.files[idx], sep='|')
            
            if len(df) > self.seq_length:
                df = df.iloc[-self.seq_length:]
            else:
                df = df.iloc[-self.seq_length:].reindex(range(self.seq_length), method='bfill')
            
            features = [c for c in df.columns if c != 'SepsisLabel']
            
            # Признаки
            x = df[features].fillna(0).values.astype(np.float32)
            x = np.nan_to_num(x, nan=0, posinf=1e6, neginf=-1e6)
            
            # 🔥 TIME EMBEDDING: номер часа (0-47)
            time_embedding = np.arange(len(x), dtype=np.float32).reshape(-1, 1) / self.seq_length
            
            # Маска пропусков
            mask = (~df[features].isna()).values.astype(np.float32)
            
            # 🔥 TIME GAP: время с последнего измерения
            time_gap = np.zeros((len(x), 1), dtype=np.float32)
            for i in range(1, len(x)):
                time_gap[i] = min(time_gap[i-1] + 1, 10) if mask[i].mean() < 0.5 else 0
            
            # Метка
            y = np.float32(df['SepsisLabel'].fillna(0).iloc[-1])
            
            # Нормализация
            if self.normalize:
                x = (x - self.mean) / (self.std + 1e-8)
                x = np.clip(x, -100, 100)
            
            # Добавляем time features к x
            x = np.concatenate([x, time_embedding, time_gap], axis=1)  # 40 + 1 + 1 = 42
            
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
        except Exception as e:
            return (
                torch.zeros(self.seq_length, 42, dtype=torch.float32),
                torch.ones(self.seq_length, 40, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32)
            )


def create_dataloaders(data_dir, seq_length=48, batch_size=32, val_split=0.2):
    dataset = PhysioNetTimeAware(data_dir=data_dir, seq_length=seq_length, max_files=5000)
    
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    print(f"📊 Всего: {n_total}, Train: {n_train}, Val: {n_val}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📈 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader
