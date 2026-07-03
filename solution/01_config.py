"""
Data Fusion Contest 2026 - Задача 1 "Страж"
Антифрод: классификация неподтверждённых операций
Метрика: PR-AUC (average_precision_score)

Подход:
1. Feature Engineering из pretrain + train + pretest
2. LightGBM / CatBoost baseline
3. ICO (Implicit Constrained Optimization) для прямой оптимизации PR-AUC
4. NN + ICO
5. Ensemble

Код рассчитан на Kaggle (2x T4 GPU, 30GB RAM)
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class CFG:
    DATA_DIR = '/kaggle/input/data-fusion-2026-task1/'  # Kaggle path
    OUTPUT_DIR = '/kaggle/working/'
    SEED = 42
    N_FOLDS = 5
    
    # LightGBM
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1,  # будем тюнить
        'n_estimators': 5000,
        'verbose': -1,
        'random_state': SEED,
        'n_jobs': -1,
    }
    
    # CatBoost
    CB_PARAMS = {
        'iterations': 5000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_seed': SEED,
        'eval_metric': 'PrecisionAt:recall=0.5',  # proxy
        'task_type': 'GPU',
        'verbose': 100,
    }
    
    # NN
    NN_EPOCHS = 30
    NN_LR = 1e-3
    NN_BATCH_SIZE = 4096
    
    # ICO
    ICO_NUM_THRESHOLDS = 20  # m для Riemann-аппроксимации PR-AUC
    ICO_TAU = 50  # correction step каждые τ итераций


# Автодетект пути к данным
if os.path.exists(CFG.DATA_DIR):
    DATA_DIR = CFG.DATA_DIR
elif os.path.exists('/Users/recara/Desktop/ods_2026/'):
    DATA_DIR = '/Users/recara/Desktop/ods_2026/'
else:
    DATA_DIR = './'

print(f"Data directory: {DATA_DIR}")
print(f"Files: {os.listdir(DATA_DIR)}")
