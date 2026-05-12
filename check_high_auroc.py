import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA')

# Признаки, которые могут вызывать leakage
TREATMENT_FEATURES = [
    'Vasopressors', 'IV Fluid', 'Antibiotics', 'Mechanical Vent',
    'Dopamine', 'Dobutamine', 'Epinephrine', 'Norepinephrine'
]

print("ПРОВЕРКА ВЫСОКОГО AUROC")
print("\n Проверка уникальности пациентов...")

patient_ids = []
for file in DATA_DIR.glob('*.psv'):
    df = pd.read_csv(file, sep='|', nrows=1)
    if 'PatientID' in df.columns:
        patient_ids.append(df['PatientID'].iloc[0])
    else:
        # Если PatientID нет, используем имя файла
        patient_ids.append(file.stem)

print(f"   Всего файлов: {len(patient_ids)}")
print(f"   Уникальных пациентов: {len(set(patient_ids))}")

if len(patient_ids) != len(set(patient_ids)):
    print(" дубликаты пациентов!")
    from collections import Counter
    dupes = [k for k, v in Counter(patient_ids).items() if v > 1]
    print(f"   Дубликаты: {dupes[:10]}...")
else:
    print("Все пациенты уникальны")

print("\n2️ Проверка timing признаков лечения...")

timing_results = []
files_with_sepsis = list(DATA_DIR.glob('*.psv'))[:100]

for file in files_with_sepsis:
    try:
        df = pd.read_csv(file, sep='|')
        
        # Найдём первый час с диагнозом
        sepsis_idx = df[df['SepsisLabel'] == 1].index
        if len(sepsis_idx) == 0:
            continue
        first_sepsis = sepsis_idx[0]
        
        for feat in TREATMENT_FEATURES:
            if feat not in df.columns:
                continue
            non_null = df[df[feat].notna()].index
            if len(non_null) == 0:
                continue
            
            first_appearance = non_null[0]
            hours_diff = first_appearance - first_sepsis
            timing_results.append({
                'feature': feat,
                'hours_after_sepsis': hours_diff,
                'file': file.name
            })
    except:
        continue

if timing_results:
    df_timing = pd.DataFrame(timing_results)
    summary = df_timing.groupby('feature')['hours_after_sepsis'].mean()
    
    print(f"   Среднее время появления признаков лечения относительно диагноза:")
    for feat, mean_h in summary.items():
        status = "ПОСЛЕ" if mean_h > 0 else "ДО"
        print(f"   {feat:<20}: {mean_h:>6.2f} ч {status}")
    
    if (summary > 0).any():
        print("\nОБНАРУЖЕНА ВОЗМОЖНАЯ УТЕЧКА!")
        print("Признаки лечения появляются ПОСЛЕ диагноза.")
        print("Рекомендация: исключить эти признаки или сдвинуть метку.")
    else:
        print("\n  Признаки лечения появляются ДО диагноза — утечки нет")
else:
    print(" Признаки лечения не найдены в выборке")
print("\n  Проверка корреляции признаков с меткой...")

correlations = []
sample_files = list(DATA_DIR.glob('*.psv'))[:50]

for file in sample_files:
    try:
        df = pd.read_csv(file, sep='|')
        if 'SepsisLabel' not in df.columns or len(df) < 10:
            continue
        
        # Берём последнее значение каждого признака
        last_values = df.iloc[-1]
        label = last_values['SepsisLabel']
        
        for col in df.columns:
            if col in ['SepsisLabel', 'PatientID', 'ICUType']:
                continue
            if pd.isna(last_values[col]):
                continue
            
            correlations.append({
                'feature': col,
                'value': last_values[col],
                'label': label,
                'is_treatment': col in TREATMENT_FEATURES
            })
    except:
        continue

if correlations:
    df_corr = pd.DataFrame(correlations)
    
    # Считаем корреляцию для каждого признака
    feat_corr = {}
    for feat in df_corr['feature'].unique():
        subset = df_corr[df_corr['feature'] == feat]
        if len(subset['value'].dropna()) > 10:
            corr = subset['value'].corr(subset['label'])
            if not pd.isna(corr):
                feat_corr[feat] = abs(corr)
    
    # Топ-10 по корреляции
    top_feats = sorted(feat_corr.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"   Топ-10 признаков по корреляции с меткой:")
    for feat, corr in top_feats:
        marker = "TREATMENT!" if feat in TREATMENT_FEATURES else ""
        print(f"   {feat:<25}: {corr:.4f}{marker}")
    
    treatment_in_top = sum(1 for f, _ in top_feats if f in TREATMENT_FEATURES)
    if treatment_in_top >= 3:
        print(f"\n   {treatment_in_top}/10 топ-признаков — признаки лечения!")
        print("   Это может объяснять высокий AUROC.")
    else:
        print(f"\nТолько {treatment_in_top}/10 топ-признаков — признаки лечения")
else:
    print("Не удалось вычислить корреляции")

