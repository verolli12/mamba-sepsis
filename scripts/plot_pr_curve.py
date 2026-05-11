#!/usr/bin/env python3
"""
Precision-Recall Curve (более информативен для дисбаланса)
"""
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys
sys.path.insert(0, '../src')
from models import create_model
from dataset import create_dataloaders

# Загрузка модели
device = torch.device('cuda')
model = create_model('lstm', input_size=40).to(device)
checkpoint = torch.load('../models/lstm_best.pt', map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Загрузка данных
_, test_loader = create_dataloaders(
    '/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA',
    seq_length=48, batch_size=32)

# Предсказания
preds, labels = [], []
with torch.no_grad():
    for x, mask, y in test_loader:
        x, mask = x.to(device), mask.to(device)
        out = torch.sigmoid(model(x, mask))
        preds.extend(out.cpu().numpy().flatten())
        labels.extend(y.numpy().flatten())

preds = np.array(preds)
labels = np.array(labels)

# PR Curve
precision, recall, thresholds = precision_recall_curve(labels, preds)
pr_auc = average_precision_score(labels, preds)

# Построение
plt.figure(figsize=(10, 8))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'LSTM (PR-AUC = {pr_auc:.4f})')
plt.axhline(y=labels.mean(), color='r', linestyle='--', linewidth=1, label=f'Baseline ({labels.mean():.4f})')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (для несбалансированных данных)', fontsize=14)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../logs/lstm_pr_curve.png', dpi=300)
plt.close()

print(f"✅ PR Curve saved: ../logs/lstm_pr_curve.png")
print(f"   PR-AUC: {pr_auc:.4f}")
print(f"   Baseline (prevalence): {labels.mean():.4f}")
print(f"   PR-AUC > Baseline: {pr_auc > labels.mean()} ✅")
