#!/usr/bin/env python3
"""Перегенерация графика из сохранённых метрик"""
import json, matplotlib.pyplot as plt
from pathlib import Path

# Пути
metrics_path = Path("/home/verolli/projects/mamba/logs/real_mamba_metrics.json")
output_path = Path("/home/verolli/projects/mamba/logs/real_mamba_training.png")

# Загрузка
with open(metrics_path) as f:
    data = json.load(f)

history = data['history']
epochs = list(range(1, len(history['val_auc']) + 1))
best_auc = data['best_auc']
best_epoch = data['best_epoch']

# Построение
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=1.5)
axes[0].plot(epochs, history['val_loss'], label='Val', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Metrics
axes[1].plot(epochs, history['val_auc'], label='AUROC', color='#2ecc71', linewidth=2, marker='o')
axes[1].plot(epochs, history['val_f1'], label='F1', color='#e74c3c', linewidth=2, marker='s')
axes[1].axhline(y=best_auc, color='#2ecc71', linestyle='--', alpha=0.5, label=f'Best: {best_auc:.4f}')
axes[1].axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ График перегенерирован: {output_path}")
