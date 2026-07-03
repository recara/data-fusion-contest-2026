"""
Этап 2: LightGBM / CatBoost Baseline
Оптимизация с proper time-based CV
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import gc


def time_based_split(event_dttms, n_splits=5):
    """
    Temporal split: train на ранних данных, val на поздних.
    Имитирует реальную постановку задачи.
    """
    sorted_idx = np.argsort(event_dttms)
    fold_size = len(sorted_idx) // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(val_start + fold_size, len(sorted_idx))
        
        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[val_start:val_end]
        splits.append((train_idx, val_idx))
    
    return splits


def train_lgb_baseline(X_train, y_train, X_test, params=None, n_folds=5, 
                        customer_ids=None, use_group_kfold=True):
    """
    LightGBM с GroupKFold по customer_id (чтобы один клиент не был в train и val).
    """
    if params is None:
        params = {
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
            'n_estimators': 3000,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
            'is_unbalance': True,
        }
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    
    if use_group_kfold and customer_ids is not None:
        kf = GroupKFold(n_splits=n_folds)
        splits = kf.split(X_train, y_train, groups=customer_ids)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(X_train, y_train)
    
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*40}")
        
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]
        
        print(f"Train: {len(X_tr)}, target=1: {y_tr.sum()}")
        print(f"Val: {len(X_val)}, target=1: {y_val.sum()}")
        
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200)
            ]
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        
        score = average_precision_score(y_val, val_pred)
        scores.append(score)
        print(f"Fold {fold+1} PR-AUC: {score:.6f}")
        
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        # Feature importance
        fi = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_,
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, fi])
        
        models.append(model)
        del X_tr, y_tr, X_val, y_val
        gc.collect()
    
    # Overall score
    valid_mask = oof_preds != 0
    overall_score = average_precision_score(y_train[valid_mask], oof_preds[valid_mask])
    
    print(f"\n{'='*60}")
    print(f"Overall OOF PR-AUC: {overall_score:.6f}")
    print(f"Mean fold PR-AUC: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    print(f"{'='*60}")
    
    # Top features
    fi_summary = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(f"\nTop 20 features:")
    for feat, imp in fi_summary.head(20).items():
        print(f"  {feat}: {imp:.1f}")
    
    return models, oof_preds, test_preds, fi_summary


def train_catboost_baseline(X_train, y_train, X_test, n_folds=5, customer_ids=None):
    """
    CatBoost с GPU.
    """
    from catboost import CatBoostClassifier, Pool
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    if customer_ids is not None:
        kf = GroupKFold(n_splits=n_folds)
        splits = kf.split(X_train, y_train, groups=customer_ids)
    else:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(X_train, y_train)
    
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]
        
        # Auto-detect categorical features
        cat_features = [i for i, col in enumerate(X_train.columns) 
                       if X_train[col].dtype in ['object', 'category', 'int32', 'int64']
                       and X_train[col].nunique() < 100]
        
        model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            eval_metric='Logloss',
            task_type='GPU',
            verbose=200,
            early_stopping_rounds=100,
            auto_class_weights='Balanced',
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_features if cat_features else None,
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        
        score = average_precision_score(y_val, val_pred)
        scores.append(score)
        print(f"Fold {fold+1} PR-AUC: {score:.6f}")
        
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        models.append(model)
        
        del X_tr, y_tr, X_val, y_val
        gc.collect()
    
    valid_mask = oof_preds != 0
    overall_score = average_precision_score(y_train[valid_mask], oof_preds[valid_mask])
    print(f"\nOverall OOF PR-AUC: {overall_score:.6f}")
    
    return models, oof_preds, test_preds
