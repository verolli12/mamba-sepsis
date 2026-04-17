#!/usr/bin/env python3
"""
Создание splits для федеративного обучения
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import json

def create_label_skew_split(data_dir, n_clients=5, alpha=0.3, seed=42):
    """Label-based Non-IID split"""
    np.random.seed(seed)
    random.seed(seed)
    
    data_path = Path(data_dir)
    files = list(data_path.glob("*.psv"))
    print(f"📂 Найдено файлов: {len(files)}")
    
    # Считаем метки
    file_labels = []
    for f in files[:5000]:  # Для скорости берём подмножество
        try:
            df = pd.read_csv(f, sep='|')
            label = int(df['SepsisLabel'].fillna(0).iloc[-1])
            file_labels.append((f, label))
        except:
            continue
    
    positive = [f for f, l in file_labels if l == 1]
    negative = [f for f, l in file_labels if l == 0]
    
    print(f"📊 Positive: {len(positive)}, Negative: {len(negative)}")
    
    # Dirichlet распределение
    clients = {i: [] for i in range(n_clients)}
    
    if positive:
        weights = np.random.dirichlet([alpha] * n_clients)
        for f in positive:
            client = np.random.choice(n_clients, p=weights)
            clients[client].append((f, 1))
    
    if negative:
        weights = np.random.dirichlet([alpha] * n_clients)
        for f in negative:
            client = np.random.choice(n_clients, p=weights)
            clients[client].append((f, 0))
    
    # Сохраняем
    output = Path("../data/fed_splits")
    output.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for cid, cfiles in clients.items():
        paths = [str(f) for f, _ in cfiles]
        labels = [l for _, l in cfiles]
        
        with open(output / f"client_{cid}.txt", 'w') as out:
            for p in paths:
                out.write(p + '\n')
        
        pos = sum(labels)
        stats[cid] = {'total': len(labels), 'positive': pos, 'ratio': pos/max(len(labels),1)}
        print(f"👤 Client {cid}: {len(labels)} файлов, {pos} positive ({100*pos/max(len(labels),1):.1f}%)")
    
    with open(output / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Splits сохранены в {output}")
    return output

if __name__ == '__main__':
    data_dir = "../data/physionet.org/files/challenge-2019/1.0.0/training/training_setA"
    create_label_skew_split(data_dir, n_clients=5, alpha=0.3)
