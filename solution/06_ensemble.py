"""
Этап 5: Ensemble и создание submission
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize


def optimize_ensemble_weights(oof_preds_list, y_true, method='scipy'):
    """
    Находим оптимальные веса для ensemble, максимизируя PR-AUC.
    
    Args:
        oof_preds_list: list of OOF predictions arrays
        y_true: true labels
        method: 'scipy' или 'grid'
    """
    n_models = len(oof_preds_list)
    
    # Filter out zero predictions (folds where predictions weren't made)
    valid_mask = np.ones(len(y_true), dtype=bool)
    for preds in oof_preds_list:
        valid_mask &= (preds != 0)
    
    y_valid = y_true[valid_mask]
    preds_valid = [p[valid_mask] for p in oof_preds_list]
    
    if method == 'scipy':
        def neg_prauc(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            blended = sum(w * p for w, p in zip(weights, preds_valid))
            return -average_precision_score(y_valid, blended)
        
        # Initial weights
        w0 = np.ones(n_models) / n_models
        
        result = minimize(neg_prauc, w0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6})
        
        best_weights = np.abs(result.x)
        best_weights = best_weights / best_weights.sum()
        best_score = -result.fun
        
    elif method == 'grid':
        # Grid search for 2-3 model ensemble
        best_score = 0
        best_weights = np.ones(n_models) / n_models
        
        if n_models == 2:
            for w1 in np.arange(0, 1.01, 0.05):
                w2 = 1 - w1
                blended = w1 * preds_valid[0] + w2 * preds_valid[1]
                score = average_precision_score(y_valid, blended)
                if score > best_score:
                    best_score = score
                    best_weights = np.array([w1, w2])
        
        elif n_models == 3:
            for w1 in np.arange(0, 1.01, 0.05):
                for w2 in np.arange(0, 1.01 - w1, 0.05):
                    w3 = 1 - w1 - w2
                    blended = w1 * preds_valid[0] + w2 * preds_valid[1] + w3 * preds_valid[2]
                    score = average_precision_score(y_valid, blended)
                    if score > best_score:
                        best_score = score
                        best_weights = np.array([w1, w2, w3])
    
    print(f"Optimal weights: {best_weights}")
    print(f"Ensemble PR-AUC: {best_score:.6f}")
    
    # Individual model scores
    for i, preds in enumerate(preds_valid):
        score = average_precision_score(y_valid, preds)
        print(f"  Model {i}: PR-AUC = {score:.6f}")
    
    return best_weights, best_score


def rank_averaging(preds_list):
    """
    Rank averaging: усредняем ранги предсказаний вместо самих предсказаний.
    Более robust к разным масштабам предсказаний.
    """
    from scipy.stats import rankdata
    
    ranks = []
    for preds in preds_list:
        ranks.append(rankdata(preds) / len(preds))
    
    return np.mean(ranks, axis=0)


def create_submission(test_event_ids, predictions, filename='submission.csv'):
    """Создаёт файл submission."""
    sub = pd.DataFrame({
        'event_id': test_event_ids,
        'predict': predictions,
    })
    sub.to_csv(filename, index=False)
    print(f"Submission saved: {filename}")
    print(f"  Shape: {sub.shape}")
    print(f"  Predict stats: mean={predictions.mean():.6f}, "
          f"std={predictions.std():.6f}, "
          f"min={predictions.min():.6f}, max={predictions.max():.6f}")
    return sub
