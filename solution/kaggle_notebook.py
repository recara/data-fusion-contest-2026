"""
=============================================================================
Data Fusion Contest 2026 - Задача 1 "Страж"
Антифрод: бинарная классификация неподтверждённых операций
Метрика: PR-AUC (average_precision_score)

Подход:
1. Feature Engineering из pretrain + train + pretest  
2. LightGBM baseline (logloss + custom PR-AUC proxy objective)
3. CatBoost baseline (GPU)
4. NN + ICO (Implicit Constrained Optimization) для прямой оптимизации PR-AUC
5. Ensemble с оптимальными весами

Запускать на Kaggle (2x T4 GPU, 30GB RAM)
=============================================================================
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIG
# ============================================================
SEED = 42
np.random.seed(SEED)

# Kaggle dataset path — ПОМЕНЯЙ на свой dataset slug!
DATA_DIR = '/kaggle/input/data-fusion-2026-task1/'
OUTPUT_DIR = '/kaggle/working/'

# Fallback для локальной отладки
if not os.path.exists(DATA_DIR):
    DATA_DIR = './'
    OUTPUT_DIR = './'

print(f"Data: {DATA_DIR}")
print(f"Files: {[f for f in os.listdir(DATA_DIR) if f.endswith('.parquet') or f.endswith('.csv')]}")


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("STEP 1: FEATURE ENGINEERING")
print("="*60)

def process_chunk_agg(df, prefix=''):
    """Агрегатные фичи по клиенту."""
    result = pd.DataFrame()
    
    # Сумма операций
    amt = df.groupby('customer_id')['operaton_amt'].agg(['count','sum','mean','std','median','min','max'])
    amt.columns = [f'{prefix}amt_{c}' for c in amt.columns]
    result = amt
    
    # Уникальные значения категорий
    for col in ['event_type_nm','event_desc','channel_indicator_type',
                'channel_indicator_sub_type','mcc_code']:
        if col in df.columns:
            result[f'{prefix}{col}_nunique'] = df.groupby('customer_id')[col].nunique()
    
    # Безопасность
    for col in ['phone_voip_call_state','web_rdp_connection']:
        if col in df.columns:
            result[f'{prefix}{col}_sum'] = df.groupby('customer_id')[col].sum()
            result[f'{prefix}{col}_mean'] = df.groupby('customer_id')[col].mean()
    
    for col in ['compromised','developer_tools']:
        if col in df.columns:
            df[f'_{col}_num'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            result[f'{prefix}{col}_sum'] = df.groupby('customer_id')[f'_{col}_num'].sum()
            result[f'{prefix}{col}_mean'] = df.groupby('customer_id')[f'_{col}_num'].mean()
    
    # Время
    if 'event_dttm' in df.columns:
        dt = pd.to_datetime(df['event_dttm'], errors='coerce')
        df['_hour'] = dt.dt.hour
        df['_dow'] = dt.dt.dayofweek
        df['_is_night'] = (df['_hour'] < 6).astype(int)
        df['_is_weekend'] = (df['_dow'] >= 5).astype(int)
        
        result[f'{prefix}night_ratio'] = df.groupby('customer_id')['_is_night'].mean()
        result[f'{prefix}weekend_ratio'] = df.groupby('customer_id')['_is_weekend'].mean()
        result[f'{prefix}hour_mean'] = df.groupby('customer_id')['_hour'].mean()
        result[f'{prefix}hour_std'] = df.groupby('customer_id')['_hour'].std()
        
        # Среднее время между операциями
        df_sorted = df.sort_values(['customer_id','event_dttm'])
        df_sorted['_dt'] = dt
        time_diffs = df_sorted.groupby('customer_id')['_dt'].apply(
            lambda x: x.diff().dt.total_seconds().mean() / 3600 if len(x) > 1 else np.nan
        )
        result[f'{prefix}avg_hours_between'] = time_diffs
        
        # Cleanup
        df.drop(columns=[c for c in df.columns if c.startswith('_')], inplace=True, errors='ignore')
    
    # Сессии, tz, OS
    for col in ['session_id','timezone','operating_system_type']:
        if col in df.columns:
            result[f'{prefix}{col}_nunique'] = df.groupby('customer_id')[col].nunique()
    
    # POS
    if 'pos_cd' in df.columns:
        result[f'{prefix}pos_cd_nunique'] = df.groupby('customer_id')['pos_cd'].nunique()
    
    return result.fillna(0)


# --- Pretrain aggregates ---
print("[1/6] Pretrain aggregates...")
pretrain_agg_parts = []
for i in range(1, 4):
    print(f"  Reading pretrain_part_{i}...")
    df = pd.read_parquet(f'{DATA_DIR}pretrain_part_{i}.parquet')
    agg = process_chunk_agg(df, prefix='pt_')
    pretrain_agg_parts.append(agg)
    del df; gc.collect()
pretrain_agg = pd.concat(pretrain_agg_parts)
del pretrain_agg_parts; gc.collect()
print(f"  Pretrain agg: {pretrain_agg.shape}")


# --- Train aggregates ---
print("\n[2/6] Train aggregates + raw data...")
train_agg_parts = []
train_raw_parts = []
for i in range(1, 4):
    print(f"  Reading train_part_{i}...")
    df = pd.read_parquet(f'{DATA_DIR}train_part_{i}.parquet')
    agg = process_chunk_agg(df, prefix='tr_')
    train_agg_parts.append(agg)
    train_raw_parts.append(df)
train_agg = pd.concat(train_agg_parts)
del train_agg_parts; gc.collect()
print(f"  Train agg: {train_agg.shape}")


# --- Combine customer agg ---
customer_agg = pretrain_agg.join(train_agg, how='outer').fillna(0)
del pretrain_agg, train_agg; gc.collect()
print(f"  Customer agg combined: {customer_agg.shape}")


# --- Labels ---
print("\n[3/6] Labels...")
labels = pd.read_parquet(f'{DATA_DIR}train_labels.parquet')
print(f"  target=1 (🔴): {(labels.target==1).sum()}")
print(f"  target=0 (🟡): {(labels.target==0).sum()}")

# Собираем train raw
train_all = pd.concat(train_raw_parts, ignore_index=True)
del train_raw_parts; gc.collect()

train_all = train_all.merge(labels[['event_id','target']], on='event_id', how='left')
train_all['target'] = train_all['target'].fillna(-1)

# Формируем трейн: 🔴 + 🟡 + sample 🟢
labeled = train_all[train_all['target'].isin([0, 1])].copy()

n_green_sample = min(200_000, int((train_all['target'] == -1).sum()))
green = train_all[train_all['target'] == -1].sample(n=n_green_sample, random_state=SEED).copy()
green['target'] = 0  # 🟢 → negative class

train_df = pd.concat([labeled, green], ignore_index=True)
del train_all, labeled, green; gc.collect()

print(f"  Train set: {len(train_df)} events")
print(f"    target=1: {int((train_df.target==1).sum())}")
print(f"    target=0: {int((train_df.target==0).sum())}")


# --- Event-level features ---
print("\n[4/6] Event-level features...")

def build_event_features(df):
    """Фичи на уровне операции."""
    feats = pd.DataFrame(index=df.index)
    
    feats['operaton_amt'] = df['operaton_amt']
    feats['log_amt'] = np.log1p(df['operaton_amt'].clip(lower=0))
    feats['event_type_nm'] = df['event_type_nm']
    feats['event_desc'] = df['event_desc']
    feats['channel_indicator_type'] = df['channel_indicator_type']
    feats['channel_indicator_sub_type'] = df['channel_indicator_sub_type']
    feats['currency_iso_cd'] = df['currency_iso_cd']
    feats['pos_cd'] = df['pos_cd']
    feats['timezone'] = df['timezone']
    feats['operating_system_type'] = df['operating_system_type']
    feats['phone_voip_call_state'] = df['phone_voip_call_state'].fillna(0)
    feats['web_rdp_connection'] = df['web_rdp_connection'].fillna(0)
    feats['compromised'] = pd.to_numeric(df['compromised'], errors='coerce').fillna(0)
    feats['developer_tools'] = pd.to_numeric(df['developer_tools'], errors='coerce').fillna(0)
    
    # Datetime
    dt = pd.to_datetime(df['event_dttm'], errors='coerce')
    feats['hour'] = dt.dt.hour
    feats['dayofweek'] = dt.dt.dayofweek
    feats['day'] = dt.dt.day
    feats['month'] = dt.dt.month
    feats['is_night'] = (feats['hour'] < 6).astype(int)
    feats['is_weekend'] = (feats['dayofweek'] >= 5).astype(int)
    
    # MCC code
    feats['mcc_code'] = pd.Categorical(df['mcc_code']).codes
    
    # Screen size
    if 'screen_size' in df.columns:
        screen = df['screen_size'].str.extract(r'(\d+)[xX×](\d+)')
        feats['screen_w'] = pd.to_numeric(screen[0], errors='coerce').fillna(0)
        feats['screen_h'] = pd.to_numeric(screen[1], errors='coerce').fillna(0)
    
    # Battery
    if 'battery' in df.columns:
        feats['battery'] = pd.to_numeric(df['battery'], errors='coerce').fillna(-1)
    
    # Language (encoded)
    if 'accept_language' in df.columns:
        feats['accept_lang'] = pd.Categorical(df['accept_language']).codes
    if 'browser_language' in df.columns:
        feats['browser_lang'] = pd.Categorical(df['browser_language']).codes
    
    return feats

train_event = build_event_features(train_df)

# --- Relative features ---
print("\n[5/6] Relative features...")

def add_relative_features(event_df, cust_agg, customer_ids):
    """Фичи: отклонение от профиля клиента."""
    # Ensure all customer_ids in agg
    missing = set(customer_ids) - set(cust_agg.index)
    if missing:
        miss_df = pd.DataFrame(0, index=list(missing), columns=cust_agg.columns)
        cust_agg = pd.concat([cust_agg, miss_df])
    
    cdata = cust_agg.loc[customer_ids].reset_index(drop=True)
    
    # Z-score суммы
    if 'tr_amt_mean' in cdata.columns and 'tr_amt_std' in cdata.columns:
        event_df['amt_zscore'] = (event_df['operaton_amt'].values - cdata['tr_amt_mean'].values) / (cdata['tr_amt_std'].values + 1e-8)
        event_df['amt_ratio_mean'] = event_df['operaton_amt'].values / (cdata['tr_amt_mean'].values + 1e-8)
        event_df['amt_ratio_max'] = event_df['operaton_amt'].values / (cdata['tr_amt_max'].values + 1e-8)
    
    # Необычное время
    if 'tr_night_ratio' in cdata.columns:
        event_df['unusual_night'] = event_df['is_night'].values * (1 - cdata['tr_night_ratio'].values)
    if 'tr_weekend_ratio' in cdata.columns:
        event_df['unusual_weekend'] = event_df['is_weekend'].values * (1 - cdata['tr_weekend_ratio'].values)
    
    return event_df, cust_agg

train_event, customer_agg = add_relative_features(
    train_event, customer_agg, train_df['customer_id'].values
)

# Customer-level features
cust_for_train = customer_agg.loc[train_df['customer_id'].values].reset_index(drop=True)
X_train = pd.concat([train_event, cust_for_train], axis=1)
y_train = train_df['target'].values.astype(int)
train_customer_ids = train_df['customer_id'].values
train_event_ids = train_df['event_id'].values

del train_event, cust_for_train
gc.collect()

print(f"  X_train: {X_train.shape}")


# --- Test features ---
print("\n[6/6] Test features...")

# Pretest agg
print("  Pretest agg...")
pretest = pd.read_parquet(f'{DATA_DIR}pretest.parquet')
pretest_agg = process_chunk_agg(pretest, prefix='ptest_')
customer_agg = customer_agg.join(pretest_agg, how='outer').fillna(0)
del pretest_agg; gc.collect()

# Также добавим pretest agg по самим тестовым клиентам из pretest
# (история операций перед тестовым днём)
pretest_event_agg = process_chunk_agg(pretest, prefix='ptest_recent_')
customer_agg = customer_agg.join(pretest_event_agg, how='outer', rsuffix='_recent').fillna(0)
del pretest, pretest_event_agg; gc.collect()

# Test
print("  Test events...")
test = pd.read_parquet(f'{DATA_DIR}test.parquet')
test_event = build_event_features(test)
test_event, customer_agg = add_relative_features(
    test_event, customer_agg, test['customer_id'].values
)

cust_for_test = customer_agg.loc[test['customer_id'].values].reset_index(drop=True)
X_test = pd.concat([test_event, cust_for_test], axis=1)
test_event_ids = test['event_id'].values

# Align columns
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

del test_event, cust_for_test, test
gc.collect()

print(f"  X_test: {X_test.shape}")
print(f"  Features: {len(common_cols)}")

# Replace inf
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 2. LIGHTGBM BASELINE
# ============================================================
print("\n" + "="*60)
print("STEP 2: LIGHTGBM BASELINE")
print("="*60)

import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'average_precision',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 5000,
    'verbose': -1,
    'random_state': SEED,
    'n_jobs': -1,
    'is_unbalance': True,
}

N_FOLDS = 5
kf = GroupKFold(n_splits=N_FOLDS)

lgb_oof = np.zeros(len(X_train))
lgb_test = np.zeros(len(X_test))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train, groups=train_customer_ids)):
    print(f"\n--- LGB Fold {fold+1}/{N_FOLDS} ---")
    
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    print(f"  Train: {len(X_tr)} (pos={y_tr.sum()}), Val: {len(X_val)} (pos={y_val.sum()})")
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    lgb_oof[val_idx] = val_pred
    
    score = average_precision_score(y_val, val_pred)
    lgb_scores.append(score)
    print(f"  Fold {fold+1} PR-AUC: {score:.6f}")
    
    lgb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    del X_tr, y_tr, X_val, y_val, model
    gc.collect()

valid_mask = lgb_oof != 0
lgb_overall = average_precision_score(y_train[valid_mask], lgb_oof[valid_mask])
print(f"\nLGB Overall OOF PR-AUC: {lgb_overall:.6f}")
print(f"LGB Mean: {np.mean(lgb_scores):.6f} ± {np.std(lgb_scores):.6f}")

# Feature importance
fi = pd.Series(model.feature_importances_, index=common_cols).sort_values(ascending=False)
print("\nTop 20 features:")
print(fi.head(20))


# ============================================================
# 3. CATBOOST BASELINE (GPU)
# ============================================================
print("\n" + "="*60)
print("STEP 3: CATBOOST BASELINE (GPU)")
print("="*60)

try:
    from catboost import CatBoostClassifier
    
    cb_oof = np.zeros(len(X_train))
    cb_test = np.zeros(len(X_test))
    cb_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train, groups=train_customer_ids)):
        print(f"\n--- CB Fold {fold+1}/{N_FOLDS} ---")
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        
        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            random_seed=SEED,
            eval_metric='Logloss',
            task_type='GPU',
            verbose=500,
            early_stopping_rounds=200,
            auto_class_weights='Balanced',
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        val_pred = model.predict_proba(X_val)[:, 1]
        cb_oof[val_idx] = val_pred
        
        score = average_precision_score(y_val, val_pred)
        cb_scores.append(score)
        print(f"  Fold {fold+1} PR-AUC: {score:.6f}")
        
        cb_test += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        del X_tr, y_tr, X_val, y_val, model
        gc.collect()
    
    valid_mask_cb = cb_oof != 0
    cb_overall = average_precision_score(y_train[valid_mask_cb], cb_oof[valid_mask_cb])
    print(f"\nCB Overall OOF PR-AUC: {cb_overall:.6f}")
    HAS_CATBOOST = True
except ImportError:
    print("CatBoost not available, skipping...")
    HAS_CATBOOST = False


# ============================================================
# 4. NN + ICO (Прямая оптимизация PR-AUC)
# ============================================================
print("\n" + "="*60)
print("STEP 4: NN + ICO (PR-AUC Direct Optimization)")
print("="*60)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Scaled data ---
scaler = StandardScaler()
X_train_np = np.nan_to_num(X_train.values.astype(np.float32), 0)
X_test_np = np.nan_to_num(X_test.values.astype(np.float32), 0)
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# --- Model ---
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))

class FraudNet(nn.Module):
    def __init__(self, input_dim, dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_dim, dims[0]), nn.BatchNorm1d(dims[0]), nn.GELU(), nn.Dropout(dropout)]
        for i in range(len(dims)-1):
            if dims[i] != dims[i+1]:
                layers.extend([nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.GELU(), nn.Dropout(dropout)])
            layers.append(ResBlock(dims[i+1], dropout))
        layers.extend([nn.Linear(dims[-1], 64), nn.GELU(), nn.Dropout(dropout/2), nn.Linear(64, 1)])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- ICO Loss ---
class ICO_PRAUC(nn.Module):
    """Implicit Constrained Optimization for PR-AUC."""
    def __init__(self, m=20, temp=1.0, recall_lo=0.1, recall_hi=1.0):
        super().__init__()
        self.m = m
        self.temp = temp
        self.beta = torch.linspace(recall_lo, recall_hi, m)
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, m))
    
    def correction(self, scores, labels):
        with torch.no_grad():
            pos_scores = torch.sort(scores[labels==1], descending=True).values
            total_pos = (labels==1).sum().float()
            for i, b in enumerate(self.beta):
                idx = min(int(b * total_pos), len(pos_scores)-1)
                if idx < len(pos_scores):
                    self.thresholds.data[i] = pos_scores[idx]
    
    def forward(self, scores, labels):
        dev = scores.device
        beta = self.beta.to(dev)
        lam = self.thresholds.to(dev)
        
        precs, gs = [], []
        for i in range(self.m):
            pos = (labels==1).float()
            neg = (labels==0).float()
            above = torch.sigmoid((scores - lam[i]) / self.temp)
            tp = (pos * above).sum()
            fp = (neg * above).sum()
            fn = (pos * (1 - above)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            precs.append(prec)
            gs.append(rec - beta[i])
        
        f = -torch.stack(precs).mean()
        
        # r_i = (∂f/∂λ_i) / (∂g_i/∂λ_i) — stop gradient
        rs = []
        for i in range(self.m):
            df_dl = torch.autograd.grad(-precs[i]/self.m, lam, retain_graph=True, create_graph=False)[0][i]
            dg_dl = torch.autograd.grad(gs[i], lam, retain_graph=True, create_graph=False)[0][i]
            rs.append((df_dl / (dg_dl + 1e-8)).detach())
        
        correction = sum(rs[i] * gs[i] for i in range(self.m))
        return f - correction

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = targets * torch.sigmoid(logits) + (1-targets) * (1 - torch.sigmoid(logits))
        w = (1 - p_t) ** self.gamma
        a = targets * self.alpha + (1-targets) * (1-self.alpha)
        return (a * w * bce).mean()

# --- Training loop ---
input_dim = X_train_scaled.shape[1]
EPOCHS = 25
BATCH_SIZE = 4096
LR = 1e-3
ICO_WARMUP = 5
ICO_TAU = 50

nn_oof = np.zeros(len(X_train))
nn_test = np.zeros(len(X_test))
nn_scores = []
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train, groups=train_customer_ids)):
    print(f"\n--- NN-ICO Fold {fold+1}/{N_FOLDS} ---")
    
    X_tr = X_train_scaled[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train_scaled[val_idx]
    y_val = y_train[val_idx]
    
    # Weighted sampling
    pos_n = (y_tr==1).sum()
    neg_n = (y_tr==0).sum()
    sw = np.where(y_tr==1, neg_n/pos_n, 1.0)
    sampler = WeightedRandomSampler(sw, len(y_tr), replacement=True)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
    
    model = FraudNet(input_dim).to(device)
    focal = FocalLoss().to(device)
    ico = ICO_PRAUC(m=20, temp=1.0).to(device)
    
    opt = torch.optim.AdamW(list(model.parameters()) + list(ico.parameters()), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.01)
    
    best_score = 0
    best_state = None
    patience_cnt = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        step = 0
        
        # ICO weight schedule
        if epoch < ICO_WARMUP:
            ico_w = 0.0
        else:
            ico_w = min(0.5, 0.1 * (epoch - ICO_WARMUP + 1))
        
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            
            l_focal = focal(logits, by)
            
            if ico_w > 0:
                scores = torch.sigmoid(logits)
                l_ico = ico(scores, by)
                loss = (1-ico_w) * l_focal + ico_w * l_ico
            else:
                loss = l_focal
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
            step += 1
            
            # Correction step
            if ico_w > 0 and step % ICO_TAU == 0:
                with torch.no_grad():
                    ico.correction(torch.sigmoid(logits), by)
        
        sched.step()
        
        # Validate
        model.eval()
        vpreds, vlabels = [], []
        with torch.no_grad():
            for bx, by in val_dl:
                bx = bx.to(device)
                vpreds.append(torch.sigmoid(model(bx)).cpu().numpy())
                vlabels.append(by.numpy())
        vpreds = np.concatenate(vpreds)
        vlabels = np.concatenate(vlabels)
        val_score = average_precision_score(vlabels, vpreds)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/step:.4f}, val_PRAUC={val_score:.6f}, ico_w={ico_w:.2f}")
        
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= 7:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    print(f"  Best PRAUC: {best_score:.6f}")
    nn_scores.append(best_score)
    
    model.load_state_dict(best_state)
    model.eval()
    
    # OOF
    with torch.no_grad():
        for s in range(0, len(X_val), BATCH_SIZE*2):
            e = min(s+BATCH_SIZE*2, len(X_val))
            chunk = torch.FloatTensor(X_val[s:e]).to(device)
            nn_oof[val_idx[s:e]] = torch.sigmoid(model(chunk)).cpu().numpy()
    
    # Test
    with torch.no_grad():
        for s in range(0, len(X_test_scaled), BATCH_SIZE*2):
            e = min(s+BATCH_SIZE*2, len(X_test_scaled))
            nn_test[s:e] += torch.sigmoid(model(X_test_tensor[s:e])).cpu().numpy() / N_FOLDS
    
    del model, focal, ico, opt, sched
    torch.cuda.empty_cache(); gc.collect()

valid_nn = nn_oof != 0
nn_overall = average_precision_score(y_train[valid_nn], nn_oof[valid_nn])
print(f"\nNN-ICO Overall OOF PR-AUC: {nn_overall:.6f}")
print(f"NN-ICO Mean: {np.mean(nn_scores):.6f} ± {np.std(nn_scores):.6f}")


# ============================================================
# 5. ENSEMBLE
# ============================================================
print("\n" + "="*60)
print("STEP 5: ENSEMBLE")
print("="*60)

from scipy.optimize import minimize as sp_minimize
from scipy.stats import rankdata

oof_list = [lgb_oof]
test_list = [lgb_test]
names = ['LGB']

if HAS_CATBOOST:
    oof_list.append(cb_oof)
    test_list.append(cb_test)
    names.append('CB')

oof_list.append(nn_oof)
test_list.append(nn_test)
names.append('NN-ICO')

# Find optimal weights
valid_mask_all = np.ones(len(y_train), dtype=bool)
for o in oof_list:
    valid_mask_all &= (o != 0)

y_v = y_train[valid_mask_all]
oofs_v = [o[valid_mask_all] for o in oof_list]

def neg_prauc(w):
    w = np.abs(w); w = w / w.sum()
    blend = sum(wi * pi for wi, pi in zip(w, oofs_v))
    return -average_precision_score(y_v, blend)

w0 = np.ones(len(oof_list)) / len(oof_list)
res = sp_minimize(neg_prauc, w0, method='Nelder-Mead', options={'maxiter': 2000})
best_w = np.abs(res.x); best_w = best_w / best_w.sum()

print(f"Optimal weights: {dict(zip(names, best_w))}")
print(f"Ensemble PR-AUC: {-res.fun:.6f}")

for name, oof_v in zip(names, oofs_v):
    print(f"  {name}: {average_precision_score(y_v, oof_v):.6f}")

# Final predictions
final_preds = sum(w * t for w, t in zip(best_w, test_list))

# Also try rank averaging
rank_preds = np.mean([rankdata(t) / len(t) for t in test_list], axis=0)
rank_blend_oof = np.mean([rankdata(o[valid_mask_all]) / valid_mask_all.sum() for o in oof_list], axis=0)
rank_score = average_precision_score(y_v, rank_blend_oof)
print(f"Rank averaging PR-AUC: {rank_score:.6f}")

# Pick best approach
if rank_score > -res.fun:
    print("Using rank averaging!")
    final_preds = rank_preds
else:
    print("Using weighted averaging!")


# ============================================================
# 6. SUBMISSION
# ============================================================
print("\n" + "="*60)
print("STEP 6: SUBMISSION")
print("="*60)

sub = pd.DataFrame({'event_id': test_event_ids, 'predict': final_preds})
sub.to_csv(f'{OUTPUT_DIR}submission.csv', index=False)
print(f"Saved: submission.csv")
print(f"Shape: {sub.shape}")
print(f"Predict: mean={final_preds.mean():.6f}, std={final_preds.std():.6f}")

# Also save individual model submissions
sub_lgb = pd.DataFrame({'event_id': test_event_ids, 'predict': lgb_test})
sub_lgb.to_csv(f'{OUTPUT_DIR}submission_lgb.csv', index=False)

sub_nn = pd.DataFrame({'event_id': test_event_ids, 'predict': nn_test})
sub_nn.to_csv(f'{OUTPUT_DIR}submission_nn_ico.csv', index=False)

if HAS_CATBOOST:
    sub_cb = pd.DataFrame({'event_id': test_event_ids, 'predict': cb_test})
    sub_cb.to_csv(f'{OUTPUT_DIR}submission_cb.csv', index=False)

print("\nDone! 🎯")
