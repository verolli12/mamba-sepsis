#!/usr/bin/env python3
"""
DataLoader для PhysioNet 2019 Sepsis Prediction
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PhysioNetSepsisDataset(Dataset):
    def __init__(self, data_dir, seq_length=48, normalize=True):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.files = list(self.data_dir.glob("*.psv"))
        
        if len(self.files) == 0:
            raise ValueError(f"No .psv files found in {data_dir}")
        
        print(f"📂 Найдено файлов: {len(self.files)}")
        
        self.normalize = normalize
        self.mean = np.zeros(40, dtype=np.float32)
        self.std = np.ones(40, dtype=np.float32)
        
        if normalize:
            self._compute_stats()
    
    def _compute_stats(self):
        print("⏳ Computing normalization statistics...")
        
        all_values = []
        valid_files = 0
        
        for i, file in enumerate(self.files[:500]):
            try:
                df = pd.read_csv(file, sep='|')
                features = [c for c in df.columns if c != 'SepsisLabel']
                
                numeric_df = df[features].select_dtypes(include=[np.number])
                
                if len(numeric_df.columns) > 0:
                    values = numeric_df.values.astype(np.float32)
                    values = np.nan_to_num(values, nan=0, posinf=1e6, neginf=-1e6)
                    all_values.append(values)
                    valid_files += 1
            except Exception as e:
                continue
        
        if len(all_values) > 0 and valid_files > 0:
            all_values = np.vstack(all_values)
            self.mean = np.nanmean(all_values, axis=0).astype(np.float32)
            self.std = np.nanstd(all_values, axis=0).astype(np.float32)
            self.std = np.clip(self.std, 1e-6, 1e6)
            print(f"  ✅ Вычислено из {valid_files} файлов")
            print(f"  Mean shape: {self.mean.shape}, Std shape: {self.std.shape}")
        else:
            print("  ⚠️  Используем дефолтные значения (mean=0, std=1)")
    
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
            
            x = df[features].fillna(0).values.astype(np.float32)
            x = np.nan_to_num(x, nan=0, posinf=1e6, neginf=-1e6)
            
            mask = (~df[features].isna()).values.astype(np.float32)
            y = np.float32(df['SepsisLabel'].fillna(0).iloc[-1])
            y = np.nan_to_num(y, nan=0, posinf=1, neginf=0)
            
            if self.normalize:
                x = (x - self.mean) / (self.std + 1e-8)
                x = np.nan_to_num(x, nan=0, posinf=10, neginf=-10)
                x = np.clip(x, -100, 100)
            
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
        except Exception as e:
            return (
                torch.zeros(self.seq_length, 40, dtype=torch.float32),
                torch.ones(self.seq_length, 40, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32)
            )


def create_dataloaders(data_dir, seq_length=48, batch_size=32,
                       val_split=0.2, test_split=0.1, normalize=True,
                       num_workers=0, seed=42, include_test=False):
    dataset = PhysioNetSepsisDataset(
        data_dir=data_dir,
        seq_length=seq_length,
        normalize=normalize
    )
    
    n_total = len(dataset)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Invalid splits: train split became non-positive")
    
    print(f"📊 Всего: {n_total}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"📈 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    if include_test:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def analyze_dataset(data_dir, output_path=None, max_files=None):
    """Быстрый аудит датасета: размеры, классы, статистики и связь признаков с target."""
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.psv")))
    if max_files is not None:
        files = files[:max_files]
    
    if len(files) == 0:
        raise ValueError(f"No .psv files found in {data_dir}")
    
    labels = []
    feature_names = None
    feature_stack = []
    
    for file in files:
        df = pd.read_csv(file, sep='|')
        if feature_names is None:
            feature_names = [c for c in df.columns if c != 'SepsisLabel']
        last_row = df.iloc[-1]
        x = last_row[feature_names].astype(np.float32).to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        y = float(np.nan_to_num(last_row.get("SepsisLabel", 0.0), nan=0.0, posinf=1.0, neginf=0.0))
        feature_stack.append(x)
        labels.append(y)
    
    X = np.vstack(feature_stack)
    y = np.array(labels, dtype=np.float32)
    pos_rate = float(y.mean()) if len(y) > 0 else 0.0
    
    feature_stats = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        feature_stats.append({
            "feature": name,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "corr_with_target": float(np.corrcoef(col, y)[0, 1]) if np.std(col) > 0 and np.std(y) > 0 else 0.0
        })
    
    report = {
        "n_files": len(files),
        "n_features": len(feature_names),
        "positive_rate": pos_rate,
        "negative_rate": float(1.0 - pos_rate),
        "feature_stats": feature_stats
    }
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report


class SyntheticSepsisBatch(Dataset):
    def __init__(self, n, seq_len, input_size, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, seq_len, input_size, generator=g, dtype=torch.float32)
        self.mask = torch.ones_like(self.x)
        self.y = torch.randint(0, 2, (n,), generator=g, dtype=torch.float32)
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        return self.x[i], self.mask[i], self.y[i]
