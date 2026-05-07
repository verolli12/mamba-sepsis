#!/usr/bin/env python3
"""Построение графика из вручную созданных данных"""
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Загрузите метрики
with open("/home/verolli/projects/mamba/logs/lstm_metrics.json") as f:
    metrics = json.load(f)

history = metrics['history']
epochs = list(range(1, len(history['val_auc']) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AUROC + F1
axes[1].plot(epochs, history['val_auc'], label='Val AUROC', color='green', marker='o')
axes[1].plot(epochs, history['val_f1'], label='Val F1', color='orange', marker='s')
axes[1].axhline(y=0.9918, color='green', linestyle='--', alpha=0.5, label='Best AUROC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Сохраните
output = Path("/home/verolli/projects/mamba/logs/lstm_training.png")
plt.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ График сохранён: {output}")
plt.close()
