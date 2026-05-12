
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_all_data(data_dir, seq_length=48, max_files=None):    
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.psv"))
    print(f" Найдено файлов: {len(files)}")
    
    if max_files:
        files = files[:max_files]
        print(f" Обрабатываем первые {max_files} файлов для теста")
    
    data_list = []
    labels_list = []
    
    for i, file in enumerate(files):
        try:
            df = pd.read_csv(file, sep='|')
            
            # Пропускаем файлы с недостаточным количеством строк
            if len(df) < seq_length:
                continue
            
            # Берём последние seq_length строк
            df = df.iloc[-seq_length:].copy()
            
            features = [c for c in df.columns if c != 'SepsisLabel']
            x = df[features].values.astype(np.float32)
            y = float(df['SepsisLabel'].iloc[-1])
            
            if x.shape == (seq_length, len(features)):
                data_list.append(x)
                labels_list.append(y)
                
        except Exception as e:
            continue
    
    
    X = np.array(data_list)
    y = np.array(labels_list)
    
    print(f" Загружено: {X.shape[0]} пациентов, {X.shape[1]} временных шагов, {X.shape[2]} признаков")
    
    return X, y, features

print("="*70)
print("ЭТАП 1: ЗАГРУЗКА И РАЗБИЕНИЕ ДАННЫХ")
print("="*70)

X, y, features = load_all_data(
    '/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA',
    seq_length=48,
    max_files=500  # Берём 500 для быстрого теста
)

# Разбиваем: 80% train, 20% остаток
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# От остатка: 50% val, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n Разбиение выполнено:")
print(f"   TRAIN: {X_train.shape[0]} пациентов (80%)")
print(f"   VAL:   {X_val.shape[0]} пациентов (10%)")
print(f"   TEST:  {X_test.shape[0]} пациентов (10%)")

# ═══════════════════════════════════════════════════════════════
# ЭТАП 2: АНАЛИЗ ДАННЫХ (только на TRAIN!)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ЭТАП 2: СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ (только TRAIN)")
print("="*70)

print(f"\n ОБЩАЯ ИНФОРМАЦИЯ:")
print(f"   Признаков: {len(features)}")
print(f"   Временной шаг: {X_train.shape[1]} часов на пациента")

# Баланс классов
sepsis_count = (y_train == 1).sum()
healthy_count = (y_train == 0).sum()
sepsis_pct = 100 * sepsis_count / len(y_train)

print(f"\n БАЛАНС КЛАССОВ (в TRAIN):")
print(f"   Без сепсиса: {healthy_count} ({100-sepsis_pct:.1f}%)")
print(f"   С сепсисом:  {sepsis_count} ({sepsis_pct:.1f}%)")

# Статистика по признакам (ТОЛЬКО на train!)
print(f"\n СТАТИСТИКА ПРИЗНАКОВ (первые 10):")
print(f"{'Признак':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Missing %'}")
print("-" * 80)

for i, feature in enumerate(features[:10]):
    feature_data = X_train[:, :, i].flatten()
    valid_data = feature_data[~np.isnan(feature_data)]
    missing_pct = 100 * (1 - len(valid_data) / len(feature_data))
    
    if len(valid_data) > 0:
        print(f"{feature:<20} {np.mean(valid_data):>11.3f} {np.std(valid_data):>11.3f} "
              f"{np.min(valid_data):>11.3f} {np.max(valid_data):>11.3f} {missing_pct:>10.1f}%")



print("\n" + "="*70)
print("ЭТАП 3: НОРМАЛИЗАЦИЯ (основано на TRAIN)")
print("="*70)

mean_train = []
std_train = []

for i in range(X_train.shape[2]):
    feature_data = X_train[:, :, i].flatten()
    valid_data = feature_data[~np.isnan(feature_data)]
    
    mean_train.append(np.mean(valid_data) if len(valid_data) > 0 else 0)
    std_train.append(np.std(valid_data) if len(valid_data) > 0 else 1)

mean_train = np.array(mean_train)
std_train = np.array(std_train)

print(f" Вычислены mean/std на TRAIN данных")

X_train_norm = (X_train - mean_train) / (std_train + 1e-8)
X_val_norm = (X_val - mean_train) / (std_train + 1e-8)
X_test_norm = (X_test - mean_train) / (std_train + 1e-8)

print(f"Нормализация применена ко всем наборам")

feature_idx = 0
train_feature = X_train_norm[:, :, feature_idx].flatten()
val_feature = X_val_norm[:, :, feature_idx].flatten()
test_feature = X_test_norm[:, :, feature_idx].flatten()

print(f"\n📊 Первый признак ({features[0]}) после нормализации:")
print(f"   TRAIN: mean={np.nanmean(train_feature):.4f}, std={np.nanstd(train_feature):.4f}")
print(f"   VAL:   mean={np.nanmean(val_feature):.4f}, std={np.nanstd(val_feature):.4f}")
print(f"   TEST:  mean={np.nanmean(test_feature):.4f}, std={np.nanstd(test_feature):.4f}")

if abs(np.nanmean(val_feature)) > 0.2 or abs(np.nanmean(test_feature)) > 0.2:
    print(" Распределения отличаются! Проверьте разбиение!")
else:
    print("Распределения похожи (как и ожидается)")

print(f"\n{'Признак':<20} {'Sepsis=0':<15} {'Sepsis=1':<15} {'Разница'}")
print("-" * 60)

for i, feature in enumerate(features[:10]):
    feature_data = X_train[:, :, i].flatten()
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


print(" АНАЛИЗ ЗАВЕРШЁН!")
