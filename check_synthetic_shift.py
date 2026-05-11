#!/usr/bin/env python3
"""
Синтетический сдвиг для проверки устойчивости моделей
Поддерживает: lstm, real_mamba, transformer
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
import json
import csv
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'src')
from models import create_model
from dataset import create_dataloaders


def apply_noise(x, sigma=0.1, seed=None):
    """Добавляет гауссовский шум"""
    if seed is not None:
        torch.manual_seed(seed)
    return x + torch.randn_like(x) * sigma


def apply_scaling(x, scale_factor=1.1):
    """Масштабирует данные"""
    return x * scale_factor


def apply_missing(x, mask, missing_rate=0.1, fill_value=0.0):
    """Создаёт случайные пропуски"""
    random_mask = torch.rand_like(x) > missing_rate
    new_mask = mask * random_mask.float()
    x_modified = x * random_mask + fill_value * (~random_mask)
    return x_modified, new_mask


def apply_time_warp(x, mask, warp_factor=0.1, seed=None):
    """Временное искажение последовательности"""
    if seed is not None:
        torch.manual_seed(seed)
    
    batch_size, seq_len, n_features = x.shape
    device = x.device
    
    warped_x = torch.zeros_like(x)
    warped_mask = torch.zeros_like(mask)
    
    for i in range(batch_size):
        new_len = int(seq_len * (1 + np.random.uniform(-warp_factor, warp_factor)))
        new_len = max(1, min(new_len, seq_len * 2))
        
        patient_data = x[i, :, :]
        patient_mask = mask[i, :, :]
        
        if new_len != seq_len:
            patient_data = torch.nn.functional.interpolate(
                patient_data.unsqueeze(0).transpose(1, 2),
                size=new_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            
            patient_mask = torch.nn.functional.interpolate(
                patient_mask.unsqueeze(0).transpose(1, 2),
                size=new_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
        
        if new_len < seq_len:
            padding = torch.zeros(seq_len - new_len, n_features, device=device)
            mask_padding = torch.zeros(seq_len - new_len, n_features, device=device)
            warped_x[i] = torch.cat([patient_data, padding], dim=0)
            warped_mask[i] = torch.cat([patient_mask, mask_padding], dim=0)
        else:
            warped_x[i] = patient_data[:seq_len]
            warped_mask[i] = patient_mask[:seq_len]
    
    return warped_x, warped_mask


def apply_mean_shift(x, shift_amount=0.5):
    """Сдвиг среднего значения"""
    return x + shift_amount


def apply_variance_shift(x, scale_var=1.5):
    """Изменение дисперсии"""
    return x * scale_var


def apply_outliers(x, outlier_rate=0.05, outlier_scale=5.0):
    """Добавление выбросов"""
    mask = torch.rand_like(x) < outlier_rate
    outliers = torch.randn_like(x) * outlier_scale
    return torch.where(mask, outliers, x)


def apply_combined_shift(x, mask, shift_level='low', seed=None):
    """Комбинированный сдвиг"""
    if seed is not None:
        torch.manual_seed(seed)
    
    if shift_level == 'low':
        x = apply_noise(x, 0.05)
        x = apply_scaling(x, 1.02)
        x, mask = apply_missing(x, mask, 0.05)
    elif shift_level == 'medium':
        x = apply_noise(x, 0.1)
        x = apply_scaling(x, 1.05)
        x, mask = apply_missing(x, mask, 0.1)
    elif shift_level == 'high':
        x = apply_noise(x, 0.2)
        x = apply_scaling(x, 1.1)
        x, mask = apply_missing(x, mask, 0.2)
    
    x = apply_mean_shift(x, 0.1)
    return x, mask


@torch.no_grad()
def evaluate_with_shift(model, test_loader, device, shift_fn, shift_name, n_repeats=3):
    """Оценивает модель с применённым сдвигом"""
    model.eval()
    all_results = []
    
    for repeat in range(n_repeats):
        all_preds, all_labels = [], []
        
        for x, mask, y in test_loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            
            result = shift_fn(x, mask) if shift_fn.__code__.co_argcount == 2 else shift_fn(x)
            
            if isinstance(result, tuple):
                x_shifted, mask_shifted = result
            else:
                x_shifted = result
                mask_shifted = mask
            
            logits = model(x_shifted, mask_shifted)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.extend(probs.flatten())
            all_labels.extend(y.cpu().numpy())
        
        auroc = roc_auc_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        all_results.append({'auroc': auroc, 'f1': f1})
    
    avg_auroc = np.mean([r['auroc'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    std_auroc = np.std([r['auroc'] for r in all_results])
    std_f1 = np.std([r['f1'] for r in all_results])
    
    return {
        'shift': shift_name,
        'auroc': avg_auroc,
        'auroc_std': std_auroc,
        'f1': avg_f1,
        'f1_std': std_f1,
        'n_samples': len(all_labels),
        'n_repeats': n_repeats
    }


def plot_results(results, baseline, output_path='shift_analysis.png'):
    """Строит графики устойчивости"""
    try:
        import matplotlib.pyplot as plt
        
        shifts = [r['shift'] for r in results]
        auroc_vals = [r['auroc'] for r in results]
        auroc_err = [r['auroc_std'] for r in results]
        f1_vals = [r['f1'] for r in results]
        f1_err = [r['f1_std'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.errorbar(shifts, auroc_vals, yerr=auroc_err, fmt='o-', capsize=5, color='blue')
        ax1.axhline(baseline['auroc'], linestyle='--', color='gray', label=f"Baseline: {baseline['auroc']:.4f}")
        ax1.set_xticklabels(shifts, rotation=45, ha='right')
        ax1.set_ylabel('AUROC')
        ax1.set_title('Устойчивость AUROC к сдвигам')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.errorbar(shifts, f1_vals, yerr=f1_err, fmt='o-', capsize=5, color='green')
        ax2.axhline(baseline['f1'], linestyle='--', color='gray', label=f"Baseline: {baseline['f1']:.4f}")
        ax2.set_xticklabels(shifts, rotation=45, ha='right')
        ax2.set_ylabel('F1 score')
        ax2.set_title('Устойчивость F1 к сдвигам')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"График сохранён: {output_path}")
        return True
    except ImportError:
        print("matplotlib не установлен, пропускаем визуализацию")
        return False
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Стресс-тест модели синтетическими сдвигами')
    parser.add_argument('--model', type=str, default='lstm', 
                        choices=['lstm', 'real_mamba', 'transformer', 'all'],
                        help='Тип модели для тестирования')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Путь к файлу модели (переопределяет стандартный)')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA',
                        help='Директория с данными')
    parser.add_argument('--batch-size', type=int, default=64, help='Размер батча')
    parser.add_argument('--seq-len', type=int, default=48, help='Длина последовательности')
    parser.add_argument('--repeats', type=int, default=3, help='Количество повторений каждого сдвига')
    parser.add_argument('--no-plot', action='store_true', help='Не строить графики')
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Директория для сохранения результатов')
    return parser.parse_args()


def get_model_config(model_name, custom_path=None):
    """Возвращает конфигурацию модели"""
    configs = {
        'lstm': {
            'path': custom_path or 'models_lstm_v2/lstm_best.pt',
            'log_dir': 'logs_lstm_v2'
        },
        'real_mamba': {
            'path': custom_path or 'models_mamba_v3/real_mamba_best.pt',
            'log_dir': 'logs_mamba_v3'
        },
        'transformer': {
            'path': custom_path or 'models_transformer_v2/transformer_best.pt',
            'log_dir': 'logs_transformer_v2'
        }
    }
    return configs.get(model_name)


def run_shift_test(model_name, model_path, log_dir, args):
    """Запускает тест сдвигов для одной модели"""
    print("=" * 70)
    print(f"СИНТЕТИЧЕСКИЙ СДВИГ - СТРЕСС-ТЕСТ: {model_name.upper()}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(model_path).exists():
        print(f"Ошибка: модель не найдена: {model_path}")
        return None
    
    print(f"\nЗагрузка модели: {model_path}")
    model = create_model(model_name, input_size=40)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Ошибка загрузки весов: {e}")
        return None
    
    model = model.to(device)
    model.eval()
    
    if not Path(args.data_dir).exists():
        print(f"Ошибка: данные не найдены: {args.data_dir}")
        return None
    
    print(f"Загрузка данных: {args.data_dir}")
    
    try:
        _, _, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            seq_length=args.seq_len,
            batch_size=args.batch_size,
            val_split=0.1,
            test_split=0.1,
            normalize=True,
            seed=args.seed,
            include_test=True
        )
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None
    
    print("\nБАЗОВАЯ ОЦЕНКА (без сдвига)")
    
    def no_shift(x, mask):
        return x, mask
    
    baseline = evaluate_with_shift(
        model, test_loader, device, no_shift, 'Baseline', n_repeats=1
    )
    print(f"AUROC: {baseline['auroc']:.4f} (+/-{baseline['auroc_std']:.4f})")
    print(f"F1:    {baseline['f1']:.4f} (+/-{baseline['f1_std']:.4f})")
    
    print("\nТЕСТЫ С СИНТЕТИЧЕСКИМИ СДВИГАМИ")
    print("-" * 70)
    
    shifts = [
        (lambda x, m: (apply_noise(x, 0.05), m), 'Noise sigma=0.05'),
        (lambda x, m: (apply_noise(x, 0.1), m), 'Noise sigma=0.1'),
        (lambda x, m: (apply_noise(x, 0.2), m), 'Noise sigma=0.2'),
        (lambda x, m: (apply_scaling(x, 1.05), m), 'Scale x1.05'),
        (lambda x, m: (apply_scaling(x, 1.1), m), 'Scale x1.1'),
        (lambda x, m: apply_missing(x, m, 0.05), 'Missing 5%'),
        (lambda x, m: apply_missing(x, m, 0.1), 'Missing 10%'),
        (lambda x, m: apply_missing(x, m, 0.2), 'Missing 20%'),
        (lambda x, m: apply_time_warp(x, m, 0.1), 'Time warp 10%'),
        (lambda x, m: (apply_mean_shift(x, 0.5), m), 'Mean shift +0.5'),
        (lambda x, m: (apply_variance_shift(x, 1.5), m), 'Variance x1.5'),
        (lambda x, m: (apply_outliers(x, 0.05, 5.0), m), 'Outliers 5%'),
        (lambda x, m: apply_combined_shift(x, m, 'low'), 'Combined LOW'),
        (lambda x, m: apply_combined_shift(x, m, 'medium'), 'Combined MEDIUM'),
        (lambda x, m: apply_combined_shift(x, m, 'high'), 'Combined HIGH'),
    ]
    
    results = [baseline]
    
    for shift_fn, shift_name in shifts:
        res = evaluate_with_shift(
            model, test_loader, device, shift_fn, shift_name,
            n_repeats=args.repeats
        )
        results.append(res)
        auroc_drop = baseline['auroc'] - res['auroc']
        f1_drop = baseline['f1'] - res['f1']
        
        if auroc_drop < 0.02:
            status = "[OK]"
        elif auroc_drop < 0.05:
            status = "[WARNING]"
        else:
            status = "[ERROR]"
        
        print(f"{status} {shift_name:20} AUROC: {res['auroc']:.4f} (+/-{res['auroc_std']:.3f}) drop={auroc_drop:.4f} | F1: {res['f1']:.4f} (+/-{res['f1_std']:.3f}) drop={f1_drop:.4f}")
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_json = output_dir / 'synthetic_shift_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    out_csv = output_dir / 'synthetic_shift_results.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Shift', 'AUROC', 'AUROC_std', 'F1', 'F1_std'])
        for r in results:
            writer.writerow([r['shift'], r['auroc'], r['auroc_std'], r['f1'], r['f1_std']])
    
    if not args.no_plot:
        plot_results(results[1:], baseline, str(output_dir / 'shift_analysis.png'))
    
    max_drop = max(baseline['auroc'] - r['auroc'] for r in results[1:])
    
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИЯ:")
    if max_drop < 0.02:
        print("   Модель ОЧЕНЬ устойчива к сдвигам")
    elif max_drop < 0.05:
        print("   Модель устойчива к сдвигам (приемлемо для публикации)")
    elif max_drop < 0.10:
        print("   Модель умеренно устойчива (требуется осторожность)")
    else:
        print("   Модель чувствительна к сдвигам (нужна доработка)")
    
    print(f"\nРезультаты сохранены в:\n  {out_json}\n  {out_csv}")
    print("=" * 70)
    
    return results


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.model == 'all':
        models_to_test = ['lstm', 'real_mamba', 'transformer']
    else:
        models_to_test = [args.model]
    
    all_results = {}
    
    for model_name in models_to_test:
        config = get_model_config(model_name, args.model_path)
        if config is None:
            print(f"Неизвестная модель: {model_name}")
            continue
        
        results = run_shift_test(
            model_name=model_name,
            model_path=config['path'],
            log_dir=config['log_dir'],
            args=args
        )
        
        if results is not None:
            all_results[model_name] = results
        
        if len(models_to_test) > 1 and model_name != models_to_test[-1]:
            print("\n\n")
    
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ УСТОЙЧИВОСТИ МОДЕЛЕЙ")
        print("=" * 70)
        print(f"{'Модель':<15} {'Max ΔAUROC':<15} {'Max ΔF1':<15} {'Статус':<10}")
        print("-" * 55)
        
        for model_name, results in all_results.items():
            baseline = results[0]
            max_auroc_drop = max(baseline['auroc'] - r['auroc'] for r in results[1:])
            max_f1_drop = max(baseline['f1'] - r['f1'] for r in results[1:])
            
            if max_auroc_drop < 0.02:
                status = "OK"
            elif max_auroc_drop < 0.05:
                status = "WARNING"
            else:
                status = "ERROR"
            
            print(f"{model_name:<15} {max_auroc_drop:<15.4f} {max_f1_drop:<15.4f} {status:<10}")
        
        print("=" * 70)


if __name__ == "__main__":
    main()