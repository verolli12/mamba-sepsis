#!/bin/bash
# run_both.sh — Обучение Mamba → Transformer (50 эпох каждая)

set -e  # Остановить при ошибке

echo "========================================"
echo "🚀 ЗАПУСК ОБУЧЕНИЯ: MAMBA → TRANSFORMER"
echo "========================================"
echo ""

# Активация окружения
source /home/verolli/venvs/mamba_ssm/bin/activate

# Переход в папку
cd /home/verolli/projects/mamba/src

# Параметры
EPOCHS=50
BATCH_SIZE=16
LR=0.00005
DATA_DIR=/home/verolli/fedmamba-project/physionet.org/files/challenge-2019/1.0.0/training/training_setA
SAVE_DIR=../models
LOG_DIR=../logs
PATIENCE=10
ACCUM_STEPS=2
GRAD_CLIP=1.0
WARMUP=100
EMA_DECAY=1.0
SEED=42

# =====================
# 1. MAMBA
# =====================
echo "========================================"
echo " ЭТАП 1/2: REAL MAMBA (${EPOCHS} эпох)"
echo "========================================"
echo "Время: ~2.5 часа"
echo ""

python train.py --model real_mamba \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --data-dir ${DATA_DIR} \
  --save-dir ${SAVE_DIR} \
  --log-dir ${LOG_DIR} \
  --patience ${PATIENCE} \
  --accum-steps ${ACCUM_STEPS} \
  --grad-clip ${GRAD_CLIP} \
  --warmup ${WARMUP} \
  --ema-decay ${EMA_DECAY} \
  --seed ${SEED}

MAMBA_STATUS=$?

if [ $MAMBA_STATUS -eq 0 ]; then
    echo ""
    echo "✅ Mamba завершена успешно!"
else
    echo ""
    echo "❌ Mamba завершилась с ошибкой (код: $MAMBA_STATUS)"
    exit $MAMBA_STATUS
fi

echo ""
echo "========================================"
echo "⏳ Пауза 10 секунд перед Transformer..."
echo "========================================"
sleep 10

# =====================
# 2. TRANSFORMER
# =====================
echo ""
echo "========================================"
echo "🟣 ЭТАП 2/2: TRANSFORMER (${EPOCHS} эпох)"
echo "========================================"
echo "Время: ~2 часа"
echo ""

python train.py --model transformer \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --data-dir ${DATA_DIR} \
  --save-dir ${SAVE_DIR} \
  --log-dir ${LOG_DIR} \
  --patience ${PATIENCE} \
  --accum-steps ${ACCUM_STEPS} \
  --grad-clip ${GRAD_CLIP} \
  --warmup ${WARMUP} \
  --ema-decay ${EMA_DECAY} \
  --seed ${SEED}

TRANSFORMER_STATUS=$?

if [ $TRANSFORMER_STATUS -eq 0 ]; then
    echo ""
    echo "✅ Transformer завершён успешно!"
else
    echo ""
    echo "❌ Transformer завершился с ошибкой (код: $TRANSFORMER_STATUS)"
    exit $TRANSFORMER_STATUS
fi

# =====================
# ФИНАЛ
# =====================
echo ""
echo "========================================"
echo "🎉 ВСЕ МОДЕЛИ ОБУЧЕНЫ!"
echo "========================================"
echo ""
echo "📊 Результаты:"
echo "   LSTM:        logs/lstm_metrics.json"
echo "   Real Mamba:  logs/real_mamba_metrics.json"
echo "   Transformer: logs/transformer_metrics.json"
echo ""
echo "📈 Графики:"
echo "   LSTM:        logs/lstm_training.png"
echo "   Real Mamba:  logs/real_mamba_training.png"
echo "   Transformer: logs/transformer_training.png"
echo ""
echo "⏱️ Общее время: ~4.5-5 часов"
echo ""
