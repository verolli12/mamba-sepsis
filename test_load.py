#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

print("🔍 Проверка данных...")

# Проверьте где данные
data_dirs = [
    Path('data/training_setA'),
    Path('training_setA'),
    Path('data/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
]

found_data = False
for data_dir in data_dirs:
    if data_dir.exists():
        files = list(data_dir.glob('*.psv'))
        print(f"✅ Найдено файлов в {data_dir}: {len(files)}")
        
        if len(files) > 0:
            df = pd.read_csv(files[0], sep='|')
            print(f"✅ Колонок: {len(df.columns)}")
            print(f"✅ Пример: {files[0].name}")
            found_data = True
        break

if not found_data:
    print("❌ Данные не найдены!")
    print("📥 Скачайте с PhysioNet или распакуйте ZIP")