import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from .model import TumorNet34Bayes
from .utils import set_seed, FocalLoss, EarlyStoppingAUC, compute_metrics, save_json

def train(
    root_dir,
    data_dir="data/processed",
    results_dir="results",
    img_size=224,
    batch_size=16,
    epochs=20,
    lr=1e-4,
    weight_decay=1e-4,
    kl_scale=1e-5,
    use_amp=True
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    DATA_DIR = os.path.join(root_dir, data_dir)
    RES_DIR  = os.path.join(root_dir, results_dir)
    CKPT_DIR = os.path.join(RES_DIR, "checkpoints")
    VIS_DIR  = os.path.join(RES_DIR, "visualizations")
    LOGS_DIR = os.path.join(RES_DIR, "logs")
    for p in [CKPT_DIR, VIS_DIR, LOGS_DIR]:
        os.makedirs(p, exist_ok=True)

    # Load arrays
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0,3,1,2)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val   = torch.tensor(X_val, dtype=torch.float32).permute(0,3,1,2)
    y_val   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)

    model = TumorNet34Bayes(dropout_p=0.4).to(device)

    # Param groups (optionally separate LR for classifier)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": lr}
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    focal = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    early = EarlyStoppingAUC(patience=6, min_delta=1e-4)

    scaler = GradScaler(enabled=use_amp)

    best_auc = -1.0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits, logvar, kl = model(xb)  
                s = torch.exp(-logvar.clamp(min=-5, max=5))
                fl = focal(logits, yb)
                loss = (s * fl + logvar.mean()) + kl_scale * kl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad(), autocast(enabled=use_amp):
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, logvar, kl = model(xb)
                s = torch.exp(-logvar.clamp(min=-5, max=5))
                fl = focal(logits, yb)
                loss = (s * fl + logvar.mean()) + kl_scale * kl
                val_loss += loss.item() * xb.size(0)
                probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
                all_probs.extend(probs.tolist())
                all_labels.extend(yb.detach().cpu().numpy().ravel().tolist())

        val_loss /= len(val_loader.dataset)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(float(auc))

        print(f"Epoch {epoch}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | AUC {auc:.4f}")

        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "tumor_resnet34_bayes_best.pth"))
            print("✅ Saved best checkpoint")

        early.step(auc)
        if early.should_stop:
            print("⏹️ Early stopping on AUC")
            break

        scheduler.step(epoch)

    save_json(os.path.join(RES_DIR, "metrics.json"), history)
    print("Done. Best AUC:", best_auc)