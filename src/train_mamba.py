#!/usr/bin/env python3
"""
Обучение классификатора на полной Mamba (RealMambaClassifier: стек Mamba + LayerNorm + residual).

Ожидаемые тензоры:
  x:    [batch, seq_len, input_size]
  mask: [batch, seq_len, input_size] (1 — наблюдение, 0 — пропуск) или None
  y:    [batch] бинарные метки (0/1), float

Запуск с синтетикой (проверка пайплайна):
  python src/train_mamba.py --dummy --epochs 2

Своим DataLoader (батчи как выше) — подставьте train_loader / val_loader вместо build_dummy_loaders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import create_model, count_parameters


class SyntheticSepsisBatch(Dataset):
    """Псевдо-данные под PhysioNet-подобные формы (только для отладки обучения)."""

    def __init__(self, n: int, seq_len: int, input_size: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, seq_len, input_size, generator=g)
        self.mask = torch.ones_like(self.x)
        self.y = torch.randint(0, 2, (n,), generator=g).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return self.x[i], self.mask[i], self.y[i]


def build_dummy_loaders(
    batch_size: int, seq_len: int, input_size: int, n_train: int, n_val: int
):
    train_ds = SyntheticSepsisBatch(n_train, seq_len, input_size, seed=1)
    val_ds = SyntheticSepsisBatch(n_val, seq_len, input_size, seed=2)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x, mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, mask)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
    acc = correct / max(total, 1)
    return total_loss / max(n_batches, 1), acc


def main():
    p = argparse.ArgumentParser(description="Train Mamba (mamba_ssm) classifier")
    p.add_argument("--dummy", action="store_true", help="синтетические данные")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--input-size", type=int, default=40)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    p.add_argument("--save", type=str, default="", help="путь для сохранения весов .pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = create_model(
        "mamba",
        input_size=args.input_size,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model.to(device)
    print(f"parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    if args.dummy:
        train_loader, val_loader = build_dummy_loaders(
            args.batch_size, args.seq_len, args.input_size, n_train=256, n_val=64
        )
    else:
        print("Укажите --dummy или реализуйте загрузку своего Dataset / csv.")
        sys.exit(1)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}"
        )

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": vars(args),
                "input_size": args.input_size,
            },
            path,
        )
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
