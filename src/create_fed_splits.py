#!/usr/bin/env python3
"""
Создание Non-IID (Label Skew) splits для федеративного обучения
с использованием Dirichlet распределения (research-grade)
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns 


def dirichlet_split_research(file_labels, n_clients=5, alpha=0.3, seed=42):
    """Разделение данных по клиентам через Dirichlet (allocation-based)"""
    
    np.random.seed(seed)

    # Разделяем по классам
    positive = [f for f, l in file_labels if l == 1]
    negative = [f for f, l in file_labels if l == 0]

    clients = {i: [] for i in range(n_clients)}

    def split_class(files, label):
        if len(files) == 0:
            return
        
        np.random.shuffle(files)

        # Генерируем пропорции
        proportions = np.random.dirichlet([alpha] * n_clients)
        counts = (proportions * len(files)).astype(int)

        # Фиксируем сумму (из-за округления)
        while counts.sum() < len(files):
            counts[np.argmax(counts)] += 1

        start = 0
        for cid, count in enumerate(counts):
            subset = files[start:start + count]
            clients[cid].extend([(f, label) for f in subset])
            start += count

    # Применяем к каждому классу
    split_class(positive, 1)
    split_class(negative, 0)

    return clients


def plot_distribution(clients, save_path=None):

    
    client_ids = []
    positives = []
    negatives = []

    for cid, data in clients.items():
        labels = [l for _, l in data]
        pos = sum(labels)
        neg = len(labels) - pos

        client_ids.append(f"C{cid}")
        positives.append(pos)
        negatives.append(neg)

    x = range(len(client_ids))

    plt.figure(figsize=(10, 6))
    plt.bar(x, positives, label="Positive (Sepsis)")
    plt.bar(x, negatives, bottom=positives, label="Negative")

    plt.xticks(x, client_ids)
    plt.xlabel("Clients")
    plt.ylabel("Samples")
    plt.title("Non-IID Data Distribution (Dirichlet α=0.3)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    
    if save_path:
        # Убедитесь что папка существует
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сохранён: {save_path}")
    
    
    try:
        plt.show()
    except:
        pass  # Игнорируем ошибку в headless режиме
    
    plt.close()  # Освобождаем память
    
def plot_heatmap(clients, save_path=None):
    """
    Heatmap: клиенты × классы
    """

    n_clients = len(clients)
    matrix = np.zeros((n_clients, 2))  # 2 класса: 0 и 1

    for cid, data in clients.items():
        labels = [l for _, l in data]
        matrix[cid, 0] = labels.count(0)
        matrix[cid, 1] = labels.count(1)

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=["Negative (Healthy)", "Positive (Sepsis)"],
        yticklabels=[f"Client {i}" for i in range(n_clients)]
    )

    plt.title("Label Distribution Across Clients (Heatmap)")
    plt.xlabel("Classes")
    plt.ylabel("Clients")

    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Heatmap сохранён: {save_path}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()  # Освобождаем память
def create_label_skew_split(data_dir, n_clients=5, alpha=0.3, seed=42, max_files=5000):
    """Основная функция создания split"""
    
    np.random.seed(seed)
    random.seed(seed)

    data_path = Path(data_dir)
    files = list(data_path.glob("*.psv"))

    print(f" Найдено файлов: {len(files)}")

    # Перемешиваем (очень важно!)
    random.shuffle(files)

    # Ограничиваем для теста
    files = files[:max_files]

    # Считываем метки
    file_labels = []
    for f in files:
        try:
            df = pd.read_csv(f, sep='|')
            label = int(df['SepsisLabel'].fillna(0).iloc[-1])
            file_labels.append((f, label))
        except Exception as e:
            print(f" Ошибка чтения {f}: {e}")
            continue

    print(f"Успешно обработано файлов: {len(file_labels)}")

    # Делим через research-метод
    clients = dirichlet_split_research(
        file_labels,
        n_clients=n_clients,
        alpha=alpha,
        seed=seed
    )

    #  Проверка пустых клиентов
    for cid in clients:
        if len(clients[cid]) == 0:
            print(f" Client {cid} пустой")

    # Сохраняем
    output = Path("../data/fed_splits")
    output.mkdir(parents=True, exist_ok=True)

    stats = {}

    for cid, cfiles in clients.items():
        paths = [str(f) for f, _ in cfiles]
        labels = [l for _, l in cfiles]

        with open(output / f"client_{cid}.txt", 'w') as out:
            for p in paths:
                out.write(p + '\n')

        pos = sum(labels)
        total = len(labels)

        stats[cid] = {
            'total': total,
            'positive': pos,
            'ratio': pos / max(total, 1)
        }

        print(f"Client {cid}: {total} файлов, {pos} positive ({100 * pos / max(total,1):.1f}%)")

    with open(output / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n Splits сохранены в {output}")

    # Визуализация
    plot_distribution(clients, save_path=output / "distribution.png")
    plot_heatmap(clients, save_path=output / "heatmap.png")

    return output


if __name__ == '__main__':
    data_dir = "/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA"

    create_label_skew_split(
        data_dir,
        n_clients=5,
        alpha=0.3,
        max_files=5000  
    )
    