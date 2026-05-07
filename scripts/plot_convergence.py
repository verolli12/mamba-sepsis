import matplotlib.pyplot as plt

# Данные из логов обучения
lstm_loss = [0.67, 0.64, 0.62]  # Fed LSTM
mamba_loss = [1.7, 1.5, 1.7]    # Fed Mamba
rounds = [1, 2, 3]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rounds, lstm_loss, 'o-', label='LSTM Fed', linewidth=2, markersize=8)
ax.plot(rounds, mamba_loss, 's-', label='Mamba Fed', linewidth=2, markersize=8)
ax.set_xlabel('Раунд', fontsize=12)
ax.set_ylabel('Train Loss', fontsize=12)
ax.set_title('Федеративное обучение: Сходимость', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('../results/plots/convergence.png', dpi=300, bbox_inches='tight')
print("✅ График сохранён: results/plots/convergence.png")
plt.show()
