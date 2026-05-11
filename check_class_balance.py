#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path

# Добавьте src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import create_dataloaders

def main():
    data_dir = '/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
    train_loader, val_loader = create_dataloaders(data_dir, seq_length=48, batch_size=256)
    
    # Соберите метки
    all_labels = []
    for _, _, y in train_loader:
        all_labels.extend(y.numpy())
    
    all_labels = np.array(all_labels)
    total = len(all_labels)
    positive = int(np.sum(all_labels))
    negative = total - positive
    
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    print("=" * 60)
    print(f"Всего примеров:     {total:,}")
    print(f"Класс 0 (Healthy):  {negative:,} ({negative/total*100:.2f}%)")
    print(f"Класс 1 (Sepsis):   {positive:,} ({positive/total*100:.2f}%)")
    print(f"\n Дисбаланс: 1:{negative//positive}")
    print(f"\n pos_weight для loss: {(negative/positive):.2f}")

if __name__ == "__main__":
    main()
