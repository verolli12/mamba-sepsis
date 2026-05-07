#!/usr/bin/env python3
"""Проверка калибровки модели (ECE - Expected Calibration Error)"""
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error — чем меньше, тем лучше"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece

# Симуляция (замените на реальные предсказания если есть)
np.random.seed(42)
y_true = np.concatenate([np.ones(600), np.zeros(3400)])
y_prob = np.concatenate([
    np.random.beta(8, 2, 600),
    np.random.beta(2, 8, 3400)
])

ece = compute_ece(y_true, y_prob)
print(f"🌡️  Expected Calibration Error (ECE): {ece:.4f}")
print(f"📊 Интерпретация:")
print(f"   ECE < 0.02: ✅ Отличная калибровка")
print(f"   ECE 0.02-0.05: 🟡 Приемлемо")
print(f"   ECE > 0.05: ⚠️  Требуется калибровка")
print(f"\n💡 Ваш результат: {'✅' if ece < 0.02 else '🟡' if ece < 0.05 else '⚠️'}")
