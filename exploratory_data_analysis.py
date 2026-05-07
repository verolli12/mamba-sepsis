# src/exploratory_data_analysis.py
#!/usr/bin/env python3
"""
Разведочный анализ данных (EDA)
ПРАВИЛЬНЫЙ порядок: РАЗБИЕНИЕ → потом АНАЛИЗ
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_data(data_dir, seq_length=48):
    """Загружаем ВСЕ данные из файлов"""
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.psv"))
    print(f"📂 Найдено файлов: {len(files)}")
    
    data_list = []
    labels_list = []
    
    for file in files[:100]:  # Берём первые 100 для теста
        try:
            df = pd.read_csv(file, sep='|')
            if len(df) > seq_length:
                df = df.iloc[-seq_length:]
            
            features = [c for c in df.columns if c != 'SepsisLabel']
            x = df[features].values.astype(np.float32)
            y = float(df['SepsisLabel'].iloc[-1])
            
            data_list.append(x)
            labels_list.append(y)
        except:
            continue
    
    return np.array(data_list), np.array(labels_list), features

# ═══════════════════════════════════════════════════════════════
# ЭТАП 1: ЗАГРУЗИТЬ И РАЗБИТЬ ДАННЫЕ (ДО статистики!)
# ═══════════════════════════════════════════════════════════════

print("="*70)
print("ЭТАП 1: ЗАГРУЗКА И РАЗБИЕНИЕ ДАННЫХ")
print("="*70)

X, y, features = load_all_data('../data/training_setA')

# Разбиваем: 80% train, 20% остаток
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# От остатка: 50% val, 50% test (то есть 10% и 10% от всех данных)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n✅ Разбиение выполнено:")
print(f"   TRAIN: {X_train.shape[0]} пациентов (80%)")
print(f"   VAL:   {X_val.shape[0]} пациентов (10%)")
print(f"   TEST:  {X_test.shape[0]} пациентов (10%)")

# ═══════════════════════════════════════════════════════════════
# ЭТАП 2: АНАЛИЗ ДАННЫХ (только на TRAIN!)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ЭТАП 2: СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ (только TRAIN)")
print("="*70)

# Общая информация
print(f"\n📊 ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"   Признаков: {len(features)}")
print(f"   Временной шаг: 48 часов на пациента")

# Баланс классов
sepsis_count = (y_train == 1).sum()
healthy_count = (y_train == 0).sum()
sepsis_pct = 100 * sepsis_count / len(y_train)

print(f"\n🏥 БАЛАНС КЛАССОВ (в TRAIN):")
print(f"   Без сепсиса: {healthy_count} ({100-sepsis_pct:.1f}%)")
print(f"   С сепсисом:  {sepsis_count} ({sepsis_pct:.1f}%)")

# Статистика по признакам (ТОЛЬКО на train!)
print(f"\n📈 СТАТИСТИКА ПРИЗНАКОВ (первые 10):")
print(f"{'Признак':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Missing %'}")
print("-" * 80)

for i, feature in enumerate(features[:10]):
    feature_data = X_train[:, :, i].flatten()  # Все часы, все пациенты
    
    # Убираем NaN
    valid_data = feature_data[~np.isnan(feature_data)]
    
    missing_pct = 100 * (1 - len(valid_data) / len(feature_data))
    
    if len(valid_data) > 0:
        print(f"{feature:<20} {np.mean(valid_data):>11.3f} {np.std(valid_data):>11.3f} "
              f"{np.min(valid_data):>11.3f} {np.max(valid_data):>11.3f} {missing_pct:>10.1f}%")

# ═══════════════════════════════════════════════════════════════
# ЭТАП 3: НОРМАЛИЗАЦИЯ (основано на TRAIN статистике)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ЭТАП 3: НОРМАЛИЗАЦИЯ (основано на TRAIN)")
print("="*70)

# Вычисляем mean/std ТОЛЬКО на TRAIN
mean_train = []
std_train = []

for i in range(X_train.shape[2]):  # Для каждого признака
    feature_data = X_train[:, :, i].flatten()
    valid_data = feature_data[~np.isnan(feature_data)]
    
    mean_train.append(np.mean(valid_data) if len(valid_data) > 0 else 0)
    std_train.append(np.std(valid_data) if len(valid_data) > 0 else 1)

mean_train = np.array(mean_train)
std_train = np.array(std_train)

print(f"✅ Вычислены mean/std на TRAIN данных")
print(f"   Mean shape: {mean_train.shape}")
print(f"   Std shape:  {std_train.shape}")

# Нормализуем ВСЕ наборы
X_train_norm = (X_train - mean_train) / (std_train + 1e-8)
X_val_norm = (X_val - mean_train) / (std_train + 1e-8)
X_test_norm = (X_test - mean_train) / (std_train + 1e-8)

print(f"\n✅ Нормализация применена ко всем наборам")

# ═══════════════════════════════════════════════════════════════
# ЭТАП 4: ПРОВЕРКА РАСПРЕДЕЛЕНИЙ (train vs val vs test)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ЭТАП 4: ПРОВЕРКА (распределения похожи?)")
print("="*70)

# Пример: первый признак
feature_idx = 0
train_feature = X_train_norm[:, :, feature_idx].flatten()
val_feature = X_val_norm[:, :, feature_idx].flatten()
test_feature = X_test_norm[:, :, feature_idx].flatten()

print(f"\nПервый признак ({features[0]}) после нормализации:")
print(f"   TRAIN: mean={np.nanmean(train_feature):.4f}, std={np.nanstd(train_feature):.4f}")
print(f"   VAL:   mean={np.nanmean(val_feature):.4f}, std={np.nanstd(val_feature):.4f}")
print(f"   TEST:  mean={np.nanmean(test_feature):.4f}, std={np.nanstd(test_feature):.4f}")

# Если отличаются ОЧЕНЬ сильно → проблема!
if abs(np.nanmean(val_feature)) > 0.2 or abs(np.nanmean(test_feature)) > 0.2:
    print("   ⚠️  ВНИМАНИЕ: распределения отличаются! Проверьте разбиение!")
else:
    print("   ✅ Распределения похожи (как и ожидается)")

# ═══════════════════════════════════════════════════════════════
# ЭТАП 5: СВЯЗЬ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ЭТАП 5: КАКИЕ ПРИЗНАКИ СВЯЗАНЫ С СЕПСИСОМ?")
print("="*70)

# Для каждого признака: средние значения при сепсисе vs без сепсиса
print(f"\n{'Признак':<20} {'Sepsis=0':<15} {'Sepsis=1':<15} {'Разница'}")
print("-" * 60)

for i, feature in enumerate(features[:10]):
    feature_data = X_train[:, :, i].flatten()
    
    # Разделяем по labels
    sepsis_vals = []
    healthy_vals = []
    
    for patient_idx in range(len(y_train)):
        patient_feature = X_train[patient_idx, :, i]
        valid = patient_feature[~np.isnan(patient_feature)]
        
        if len(valid) > 0:
            if y_train[patient_idx] == 1:
                sepsis_vals.extend(valid)
            else:
                healthy_vals.extend(valid)
    
    if len(sepsis_vals) > 0 and len(healthy_vals) > 0:
        mean_healthy = np.mean(healthy_vals)
        mean_sepsis = np.mean(sepsis_vals)
        diff = abs(mean_sepsis - mean_healthy)
        
        print(f"{feature:<20} {mean_healthy:>14.3f} {mean_sepsis:>14.3f} {diff:>14.3f}")

print("\n" + "="*70)
print("✅ АНАЛИЗ ЗАВЕРШЁН!")
print("="*70)