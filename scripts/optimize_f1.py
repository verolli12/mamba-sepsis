#!/usr/bin/env python3
"""Оптимизация порога для максимизации F1 — ИСПРАВЛЕННАЯ ВЕРСИЯ"""
import json, numpy as np, torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
sys.path.insert(0, '../src')
from dataset import create_dataloaders
from models import create_model

def load_model_checkpoint(model_name, device='cuda'):
    """Загрузка модели из чекпоинта с метаданными"""
    model = create_model(model_name, input_size=40).to(device)
    
    checkpoint_path = Path(f'../models/{model_name}_best.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    #  FIX: Чекпоинт содержит словарь с метаданными
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Если сохранён просто state_dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def optimize_threshold_for_model(model_name, device='cuda'):
    """Оптимизация порога для данной модели"""
    print(f"\n🔍 Оптимизация порога для {model_name.upper()}...")
    
    # Загрузка модели
    model = load_model_checkpoint(model_name, device)
    
    # Загрузка валидационных данных
    _, val_loader = create_dataloaders(
        '/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA',
        seq_length=48, batch_size=32)
    
    # Получение предсказаний
    preds, labels = [], []
    with torch.no_grad():
        for x, mask, y in val_loader:
            x, mask = x.to(device), mask.to(device)
            out = torch.sigmoid(model(x, mask))
            preds.extend(out.cpu().numpy())
            labels.extend(y.numpy())
    
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    
    # F1 при стандартном пороге 0.5
    f1_default = f1_score(labels, (preds > 0.5).astype(int), zero_division=0)
    
    # 🔍 Поиск оптимального порога по F1
    best_t, best_f1 = 0.5, f1_default
    for t in np.linspace(0.1, 0.9, 200):  # Более тонкий поиск
        f1 = f1_score(labels, (preds > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    # 🔍 Также найдём порог для максимального Recall при Precision >= 0.90
    best_t_clinical, best_f1_clinical = 0.5, 0
    for t in np.linspace(0.1, 0.9, 200):
        prec = precision_score(labels, (preds > t).astype(int), zero_division=0)
        if prec >= 0.90:  # Клиническое требование: минимум 90% точности
            recall = recall_score(labels, (preds > t).astype(int), zero_division=0)
            f1_clinical = f1_score(labels, (preds > t).astype(int), zero_division=0)
            if f1_clinical > best_f1_clinical:
                best_f1_clinical, best_t_clinical = f1_clinical, t
    
    # Результаты
    print(f" {model_name.upper()}")
    print(f"   Стандартный порог (0.50): F1 = {f1_default:.4f}")
    print(f" Оптимальный по F1: порог = {best_t:.3f} → F1 = {best_f1:.4f}")
    print(f" Улучшение: +{best_f1 - f1_default:.4f}")
    
    if best_t_clinical != 0.5:
        print(f"Клинический порог (Precision≥90%): порог = {best_t_clinical:.3f} → F1 = {best_f1_clinical:.4f}")
    
    # Детальные метрики при оптимальном пороге
    print(f"\n Метрики при пороге {best_t:.3f}:")
    print(f"   Precision: {precision_score(labels, (preds>best_t).astype(int), zero_division=0):.4f}")
    print(f"   Recall:    {recall_score(labels, (preds>best_t).astype(int), zero_division=0):.4f}")
    print(f"   F1:        {best_f1:.4f}")
    
    # Сохранение результатов
    results = {
        'model': model_name,
        'default_threshold': 0.5,
        'default_f1': float(f1_default),
        'optimal_threshold': float(best_t),
        'optimal_f1': float(best_f1),
        'clinical_threshold': float(best_t_clinical) if best_t_clinical != 0.5 else None,
        'clinical_f1': float(best_f1_clinical) if best_t_clinical != 0.5 else None,
        'improvement': float(best_f1 - f1_default)
    }
    
    output_path = Path(f'../models/{model_name}_optimal_threshold.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nСохранено: {output_path}")
    
    return results

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Оптимизируйте нужную модель:
    # optimize_threshold_for_model('lstm', device)
    # optimize_threshold_for_model('real_mamba', device)
    
    # Или обе:
    for model in ['lstm', 'real_mamba']:
        try:
            optimize_threshold_for_model(model, device)
        except FileNotFoundError:
            print(f" Модель {model} не найдена, пропускаем...")
        except Exception as e:
            print(f" Ошибка для {model}: {e}")
    
    print("\n Оптимизация завершена!")
