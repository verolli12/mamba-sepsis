import pandas as pd
import numpy as np
from pathlib import Path


TREATMENT_FEATURES = [
    'Vasopressors',      # Вазопрессоры
    'IV Fluid',          # Внутривенная жидкость  
    'Antibiotics',       # Антибиотики
    'Mechanical Vent',   # ИВЛ
    'Dopamine',          # Дофамин
    'Dobutamine',        # Добутамин
    'Epinephrine',       # Эпинефрин
    'Norepinephrine',    # Норэпинефрин
]

# Ранние маркеры сепсиса (должны быть ДО диагноза → безопасно)
EARLY_MARKERS = [
    'Lactate',           # Лактат
    'SOFA_score',        # SOFA (если есть)
    'Temp',              # Температура
    'HR',                # Пульс
    'MAP',               # Среднее АД
    'WBC',               # Лейкоциты
    'Platelets',         # Тромбоциты
    'BUN',               # Мочевина
    'Creatinine',        # Креатинин
    'Bilirubin',         # Билирубин
]

def check_feature_timing(data_dir, n_files=200):
    
    data_dir = Path(data_dir)
    files = list(data_dir.glob('*.psv'))[:n_files]
    
    print(f"Анализируем {len(files)} файлов из {data_dir}")
    
    timing_results = []
    
    for i, file in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Обработано: {i+1}/{len(files)}")
            
        try:
            df = pd.read_csv(file, sep='|')
            
            # Найдём первый час с диагнозом "сепсис"
            sepsis_rows = df[df['SepsisLabel'] == 1]
            if len(sepsis_rows) == 0:
                continue  # Пациент без сепсиса — пропускаем
            
            first_sepsis_idx = sepsis_rows.index[0]
            
            # Для каждого признака найдём первое непустое значение
            for col in df.columns:
                if col in ['SepsisLabel', 'PatientID', 'ICUType']:
                    continue
                    
                # Найдём все непустые значения
                non_null_mask = df[col].notna()
                non_null_indices = df[non_null_mask].index
                
                if len(non_null_indices) == 0:
                    continue  # Признак полностью пустой
                
                first_appearance_idx = non_null_indices[0]
                
                # Разница в часах: отрицательная = до сепсиса, положительная = после
                hours_relative_to_sepsis = first_appearance_idx - first_sepsis_idx
                
                timing_results.append({
                    'file': file.name,
                    'feature': col,
                    'hours_relative_to_sepsis': hours_relative_to_sepsis,
                    'is_treatment': col in TREATMENT_FEATURES,
                    'is_early_marker': col in EARLY_MARKERS,
                })
                
        except Exception as e:
            print(f"Ошибка в файле {file.name}: {e}")
            continue
    
    return pd.DataFrame(timing_results)

def analyze_timing(df_timing):
    """Анализирует результаты и выводит подозрительные признаки"""
    
    print("\n" + "="*80)
    print("АНАЛИЗ ВРЕМЕННОЙ ПОСЛЕДОВАТЕЛЬНОСТИ ПРИЗНАКОВ")
    print("="*80)
    
    summary = df_timing.groupby('feature').agg({
        'hours_relative_to_sepsis': ['mean', 'median', 'std', 'min', 'max'],
        'is_treatment': 'first',
        'is_early_marker': 'first'
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    summary = summary.sort_values('hours_relative_to_sepsis_mean')
    
    print(f"\nСводная статистика по {len(summary)} признакам:")
    print(f"{'Признак':<25} {'Среднее':>10} {'Медиана':>10} {'Min':>8} {'Max':>8} {'Тип':<12}")
    print("-"*80)
    
    for _, row in summary.iterrows():
        feature = row['feature'][:24]
        mean_h = row['hours_relative_to_sepsis_mean']
        median_h = row['hours_relative_to_sepsis_median']
        min_h = row['hours_relative_to_sepsis_min']
        max_h = row['hours_relative_to_sepsis_max']
        
        # Определяем тип
        if row['is_treatment_first']:
            feature_type = "Лечение"
        elif row['is_early_marker_first']:
            feature_type = "Маркер"
        else:
            feature_type = "Другое"
        
        # Подсветка подозрительных
        if mean_h > 0 and row['is_treatment_first']:
            print(f"{feature:<25} {mean_h:>10.2f} {median_h:>10.2f} {min_h:>8.0f} {max_h:>8.0f} {feature_type}")
        else:
            print(f"   {feature:<25} {mean_h:>10.2f} {median_h:>10.2f} {min_h:>8.0f} {max_h:>8.0f} {feature_type}")
    
    # 🔍 Проверка на leakage
    print("\n" + "="*80)
    print("ПРОВЕРКА НА LABEL LEAKAGE")
    print("="*80)
    
    # Признаки лечения, которые появляются в среднем ПОСЛЕ диагноза
    suspicious = summary[
        (summary['is_treatment_first'] == True) & 
        (summary['hours_relative_to_sepsis_mean'] > 0)
    ]
    
    if len(suspicious) > 0:
        print(f"\nНАЙДЕНО {len(suspicious)} признаков лечения, появляющихся ПОСЛЕ диагноза:")
        for _, row in suspicious.iterrows():
            print(f"{row['feature']}: в среднем +{row['hours_relative_to_sepsis_mean']:.1f} часов")
        print("\nРекомендация: исключить эти признаки или сдвинуть метку во времени!")
    else:
        print("\nПризнаки лечения в среднем появляются ДО диагноза — риск leakage низкий")
    
    # Ранние маркеры, которые появляются ПОСЛЕ диагноза (странно!)
    late_markers = summary[
        (summary['is_early_marker_first'] == True) & 
        (summary['hours_relative_to_sepsis_mean'] > 5)  # >5 часов после — подозрительно
    ]
    
    if len(late_markers) > 0:
        print(f"\nНАЙДЕНО {len(late_markers)} ранних маркеров, появляющихся ПОЗДНО:")
        for _, row in late_markers.iterrows():
            print(f"{row['feature']}: в среднем +{row['hours_relative_to_sepsis_mean']:.1f} часов")
    

    print("\n" + "="*80)
    print("РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ")
    print("="*80)
    
    categories = {
        'Лечение (Treatment)': df_timing[df_timing['is_treatment'] == True],
        'Ранние маркеры (Early)': df_timing[df_timing['is_early_marker'] == True],
        'Другие признаки': df_timing[~df_timing['is_treatment'] & ~df_timing['is_early_marker']]
    }
    
    for name, data in categories.items():
        if len(data) > 0:
            mean_val = data['hours_relative_to_sepsis'].mean()
            median_val = data['hours_relative_to_sepsis'].median()
            print(f"{name:<30} Среднее: {mean_val:>7.2f} ч, Медиана: {median_val:>7.2f} ч")
    
    return summary

if __name__ == "__main__":
    # ПУТЬ К ВАШИМ ДАННЫМ:
    DATA_DIR = '/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA'
    
    print("Проверка на label leakage в PhysioNet 2019")
    print(f"Данные: {DATA_DIR}")
    
    # Запускаем анализ
    timing_df = check_feature_timing(DATA_DIR, n_files=200)
    
    if len(timing_df) == 0:
        print("Не удалось проанализировать данные")
    else:
        # Анализируем результаты
        summary = analyze_timing(timing_df)
        
        # Сохраняем результаты
        output_path = Path('leakage_check_results.csv')
        summary.to_csv(output_path, index=False)
        print(f"\n Результаты сохранены: {output_path}")
