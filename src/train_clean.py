#!/usr/bin/env python3
"""Clean training script - guaranteed to work"""
import argparse, sys, time, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from models import create_model, count_parameters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-dir', type=str, default='../data/training_setA')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seq-len', type=int, default=48)
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--save-dir', type=str, default='../models')
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model = create_model(args.model, input_size=40).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    if args.dummy:
        from dataset import SyntheticSepsisBatch
        train_ds = SyntheticSepsisBatch(256, args.seq_len, 40)
        val_ds = SyntheticSepsisBatch(64, args.seq_len, 40)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    else:
        from dataset import create_dataloaders
        train_loader, val_loader = create_dataloaders(args.data_dir, args.seq_len, args.batch_size)

    all_y = [y.numpy() for _, _, y in train_loader]
    pos = np.mean(np.concatenate(all_y))
    pos_weight = torch.tensor([(1-pos)/max(pos,1e-6)], device=device)
    print(f"Positive weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_auc, no_improve = 0, 0
    history = {'loss': [], 'auc': [], 'f1': []}
    print(f"\n🚀 Training: {args.epochs} epochs")
    print("="*60)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        for x, mask, y in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, mask)
            loss = criterion(out, y)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
        
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, mask, y in val_loader:
                out = torch.sigmoid(model(x.to(device), mask.to(device)))
                preds.extend(out.cpu().numpy())
                labels.extend(y.numpy())
        
        preds, labels = np.array(preds), np.array(labels)
        auc = roc_auc_score(labels, preds) if len(np.unique(labels))>1 else 0.5
        f1 = f1_score(labels, (preds>0.5).astype(int), zero_division=0)
        train_loss /= max(1, len(train_loader))
        dt = time.time() - t0
        
        history['loss'].append(float(train_loss))
        history['auc'].append(float(auc))
        history['f1'].append(float(f1))
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | loss={train_loss:.4f} | auc={auc:.4f} | f1={f1:.4f} | {dt:.1f}s")
        
        if auc > best_auc:
            best_auc, best_f1, best_epoch = auc, f1, epoch+1
            no_improve = 0
            torch.save(model.state_dict(), f"{args.save_dir}/{args.model}_best.pt")
            print(f"  ✅ New best!")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break

    print("="*60)
    print(f"🏆 Best AUROC: {best_auc:.4f} (epoch {best_epoch})")
    json.dump({'model': args.model, 'best_auc': float(best_auc), 'best_f1': float(best_f1), 'history': history}, 
              open(f"{args.save_dir}/{args.model}_metrics.json", 'w'), indent=2)
    print("🎉 DONE!")

if __name__ == "__main__":
    main()
