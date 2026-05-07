#!/usr/bin/env python3
"""Простая калибровка через Platt Scaling"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def calibrate_probs(y_true, y_prob):
    """Platt scaling для калибровки вероятностей"""
    # Конвертируем в logits
    y_prob = np.clip(y_prob, 1e-7, 1-1e-7)
    logits = np.log(y_prob / (1 - y_prob))
    
    # Обучаем логистическую регрессию
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y_true)
    
    return lr

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_conf = y_prob[in_bin].mean()
            avg_acc = y_true[in_bin].mean()
            ece += abs(avg_acc - avg_conf) * prop_in_bin
    return ece

# Симуляция данных (замените на реальные предсказания)
np.random.seed(42)
y_true = np.concatenate([np.ones(600), np.zeros(3400)])
y_prob_raw = np.concatenate([
    np.random.beta(8, 2, 600),
    np.random.beta(2, 8, 3400)
])

# До калибровки
ece_before = compute_ece(y_true, y_prob_raw)
print(f"📊 До калибровки: ECE = {ece_before:.4f}")

# Калибровка
calibrator = calibrate_probs(y_true, y_prob_raw)
logits = np.log(np.clip(y_prob_raw, 1e-7, 1-1e-7) / (1 - np.clip(y_prob_raw, 1e-7, 1-1e-7)))
y_prob_calibrated = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

# После калибровки
ece_after = compute_ece(y_true, y_prob_calibrated)
print(f"📊 После калибровки: ECE = {ece_after:.4f}")
print(f"✅ Улучшение ECE: {ece_before - ece_after:.4f}")

# Сохраните калибратор
import joblib
joblib.dump(calibrator, '../models/mamba_calibrator.pkl')
print("✅ Калибратор сохранён: ../models/mamba_calibrator.pkl")
