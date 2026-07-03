"""
Этап 4: Neural Network с ICO для прямой оптимизации PR-AUC
Архитектура: MLP с residual connections для табулярных данных
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
import gc


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.act(x + self.dropout(self.block(x)))


class AntiFraudMLP(nn.Module):
    """
    MLP с residual connections для табулярных данных.
    Выходит скор (logit), без sigmoid — для ICO loss.
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        blocks = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            blocks.append(ResidualBlock(hidden_dims[i + 1], dropout))
        
        self.backbone = nn.Sequential(*blocks)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)


def train_nn_ico(X_train, y_train, X_test, n_folds=5, customer_ids=None,
                 hidden_dims=[512, 256, 128], epochs=30, lr=1e-3, batch_size=4096,
                 ico_warmup=5, num_thresholds=20, temperature=1.0, 
                 ico_correction_tau=50, device='cuda'):
    """
    Обучение NN с ICO loss для прямой оптимизации PR-AUC.
    """
    from solution.s04_ico_loss import CombinedLoss
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    
    # Fill NaN
    X_train_np = np.nan_to_num(X_train_np, 0)
    X_test_np = np.nan_to_num(X_test_np, 0)
    
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    input_dim = X_train_scaled.shape[1]
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    if customer_ids is not None:
        kf = GroupKFold(n_splits=n_folds)
        splits = list(kf.split(X_train, y_train, groups=customer_ids))
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(kf.split(X_train, y_train))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"NN-ICO Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        X_tr = X_train_scaled[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train_scaled[val_idx]
        y_val = y_train[val_idx]
        
        # Weighted sampler for imbalanced data
        pos_count = (y_tr == 1).sum()
        neg_count = (y_tr == 0).sum()
        sample_weights = np.where(y_tr == 1, neg_count / pos_count, 1.0)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y_tr),
            replacement=True
        )
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_tr), 
            torch.FloatTensor(y_tr)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=sampler, num_workers=2, pin_memory=True
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, 
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # Model
        model = AntiFraudMLP(input_dim, hidden_dims).to(device)
        
        # Loss: Combined Focal + ICO
        criterion = CombinedLoss(
            num_thresholds=num_thresholds,
            temperature=temperature,
            focal_alpha=0.25,
            focal_gamma=2.0,
            ico_weight_start=0.0,
            ico_weight_end=0.5,
            warmup_epochs=ico_warmup,
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.ico.parameters()),
            lr=lr, weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        
        best_score = 0
        best_model_state = None
        patience = 7
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            criterion.set_epoch(epoch)
            
            epoch_loss = 0
            epoch_info = {}
            step = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                logits = model(batch_X)
                loss, info = criterion(logits, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                for k, v in info.items():
                    epoch_info[k] = epoch_info.get(k, 0) + v
                
                step += 1
                
                # ICO correction step
                if step % ico_correction_tau == 0 and epoch >= ico_warmup:
                    with torch.no_grad():
                        scores = torch.sigmoid(logits)
                        criterion.ico.correction_step(scores, batch_y)
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    logits = model(batch_X)
                    probs = torch.sigmoid(logits)
                    val_preds.append(probs.cpu().numpy())
                    val_labels.append(batch_y.numpy())
            
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_score = average_precision_score(val_labels, val_preds)
            
            avg_loss = epoch_loss / max(step, 1)
            n_info = {k: v / max(step, 1) for k, v in epoch_info.items()}
            
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"val_PR-AUC={val_score:.6f}, "
                  f"ico_w={n_info.get('ico_weight', 0):.2f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
            
            if val_score > best_score:
                best_score = val_score
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"  Best val PR-AUC: {best_score:.6f}")
        fold_scores.append(best_score)
        
        # Load best model
        model.load_state_dict(best_model_state)
        model.eval()
        
        # OOF predictions
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val).to(device)
            for start in range(0, len(X_val), batch_size * 2):
                end = min(start + batch_size * 2, len(X_val))
                chunk = val_tensor[start:end]
                preds = torch.sigmoid(model(chunk)).cpu().numpy()
                oof_preds[val_idx[start:end]] = preds
        
        # Test predictions
        with torch.no_grad():
            for start in range(0, len(X_test_scaled), batch_size * 2):
                end = min(start + batch_size * 2, len(X_test_scaled))
                chunk = X_test_tensor[start:end]
                preds = torch.sigmoid(model(chunk)).cpu().numpy()
                test_preds[start:end] += preds / n_folds
        
        del model, criterion, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nMean fold PR-AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    valid_mask = oof_preds != 0
    if valid_mask.sum() > 0:
        overall = average_precision_score(y_train[valid_mask], oof_preds[valid_mask])
        print(f"Overall OOF PR-AUC: {overall:.6f}")
    
    return oof_preds, test_preds
