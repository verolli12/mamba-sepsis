import matplotlib.pyplot as plt
import numpy as np

# Данные
models = ['LSTM', 'Transformer', 'Real Mamba', 'GRU-D']
auroc = [0.9908, 0.9901, 0.9788, 0.5000]
params = [219265, 798465, 480513, 164689]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

# График AUROC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# AUROC бар
bars1 = ax1.bar(models, auroc, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax1.set_title('Сравнение AUROC моделей', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim([0, 1.05])
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Случайное (0.5)')
ax1.legend()

# Добавление значений на столбцы
for bar, val in zip(bars1, auroc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Параметры бар
bars2 = ax2.bar(models, params, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Количество параметров', fontsize=12, fontweight='bold')
ax2.set_title('Размер моделей', fontsize=14, fontweight='bold', pad=20)

# Добавление значений
for bar, val in zip(bars2, params):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
             f'{val:,}', ha='center', va='bottom', fontsize=9, rotation=45)

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
print("✅ График сохранён: results_comparison.png")
plt.show()
