
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path('/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
files = sorted(list(data_dir.glob('*.psv')))

print(f"Найдено файлов: {len(files)}")

all_data = []
all_labels = []
loaded_count = 0

for i, file in enumerate(files):
    if (i + 1) % 2000 == 0:
        print(f"{i+1}/{len(files)}")
    
    try:
        df = pd.read_csv(file, sep='|')
        features = [c for c in df.columns if c != 'SepsisLabel']
        x = df[features].values.astype(np.float32)
        y = float(df['SepsisLabel'].iloc[-1])
        all_data.append(x)
        all_labels.append(y)
        loaded_count += 1
    except:
        continue

all_data = np.vstack(all_data)
all_labels = np.array(all_labels)

print(f"Загружено: {all_data.shape[0]} пациентов, {all_data.shape[1]} признаков")

mean_all = np.nanmean(all_data, axis=0)
std_all = np.nanstd(all_data, axis=0)

print(f"\nMEAN/STD на ВСЕХ {all_data.shape[0]} строках:")
print(f"  Mean[0:5]: {mean_all[:5]}")
print(f"  Std[0:5]:  {std_all[:5]}")

print(f"\nРазбиваю данные 80/10/10...")

indices = np.arange(len(all_labels))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=all_labels
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=all_labels[temp_idx]
)

print(f"Разбиение выполнено:")
print(f"   TRAIN: {len(train_idx)} ({100*len(train_idx)/len(all_labels):.0f}%)")
print(f"   VAL:   {len(val_idx)} ({100*len(val_idx)/len(all_labels):.0f}%)")
print(f"   TEST:  {len(test_idx)} ({100*len(test_idx)/len(all_labels):.0f}%)")


train_data = all_data[train_idx]
mean_train = np.nanmean(train_data, axis=0)
std_train = np.nanstd(train_data, axis=0)

print(f"\nMEAN/STD ТОЛЬКО на TRAIN {train_data.shape[0]} строках:")
print(f"  Mean[0:5]: {mean_train[:5]}")
print(f"  Std[0:5]:  {std_train[:5]}")

print("\n" + "="*70)
print("СРАВНЕНИЕ: mean/std на ALL vs на TRAIN")
print("="*70)

diff_mean = np.abs(mean_all - mean_train).mean()
diff_std = np.abs(std_all - std_train).mean()

print(f"\nСредняя разница в mean: {diff_mean:.6f}")
print(f"Средняя разница в std:  {diff_std:.6f}")

if diff_mean < 0.1:
    print("\nРазницы маленькие")
    print("   (train = 80% всех данных, поэтому похожа статистика)")
else:
    print("\nРазницы БОЛЬШИЕ → что-то не так")

print("\n" + "="*70)
print("БАЛАНС КЛАССОВ")
print("="*70)

train_labels = all_labels[train_idx]
val_labels = all_labels[val_idx]
test_labels = all_labels[test_idx]

train_sepsis_pct = 100 * (train_labels == 1).sum() / len(train_labels)
val_sepsis_pct = 100 * (val_labels == 1).sum() / len(val_labels)
test_sepsis_pct = 100 * (test_labels == 1).sum() / len(test_labels)

print(f"\nTRAIN: {(train_labels == 1).sum():5d} с сепсисом из {len(train_labels):5d} ({train_sepsis_pct:5.2f}%)")
print(f"VAL:   {(val_labels == 1).sum():5d} с сепсисом из {len(val_labels):5d} ({val_sepsis_pct:5.2f}%)")
print(f"TEST:  {(test_labels == 1).sum():5d} с сепсисом из {len(test_labels):5d} ({test_sepsis_pct:5.2f}%)")

print("\n" + "="*70)
print("ИТОГ")
print("="*70)

diff_val = abs(train_sepsis_pct - val_sepsis_pct)
diff_test = abs(train_sepsis_pct - test_sepsis_pct)

if diff_val < 1 and diff_test < 1:
    print("Баланс классов одинаковый во всех наборах!")
    print("Разделение выполнено правильно ")
else:
    print("Баланс классов отличается")
    print("Возможна проблема в разбиении")

print("="*70 + "\n")
