"""
Этап 3: ICO (Implicit Constrained Optimization) для прямой оптимизации PR-AUC
Реализация по статье Kumar et al. 2021

Ключевая идея:
PR-AUC ≈ (1/m) Σ precision(s_θ, λ_i) при recall(s_θ, λ_i) = β_i
Пороги λ_i выражаем как неявную функцию параметров модели через IFT.
Используем сигмоиду как surrogate для индикаторной функции.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SurrogateMetrics:
    """
    Дифференцируемые surrogates для TP, FP, FN, TN.
    Используем сигмоиду вместо step function.
    """
    
    @staticmethod
    def soft_indicator_gt(x, threshold, temperature=1.0):
        """σ((x - threshold) / temperature) — soft indicator I(x > threshold)"""
        return torch.sigmoid((x - threshold) / temperature)
    
    @staticmethod
    def soft_indicator_lt(x, threshold, temperature=1.0):
        """σ((threshold - x) / temperature) — soft indicator I(x < threshold)"""
        return torch.sigmoid((threshold - x) / temperature)
    
    @staticmethod
    def soft_tp(scores, labels, threshold, temperature=1.0):
        """Soft True Positives: Σ_{y=1} σ(score - threshold)"""
        pos_mask = (labels == 1).float()
        return (pos_mask * SurrogateMetrics.soft_indicator_gt(scores, threshold, temperature)).sum()
    
    @staticmethod
    def soft_fp(scores, labels, threshold, temperature=1.0):
        """Soft False Positives: Σ_{y=0} σ(score - threshold)"""
        neg_mask = (labels == 0).float()
        return (neg_mask * SurrogateMetrics.soft_indicator_gt(scores, threshold, temperature)).sum()
    
    @staticmethod
    def soft_fn(scores, labels, threshold, temperature=1.0):
        """Soft False Negatives: Σ_{y=1} σ(threshold - score)"""
        pos_mask = (labels == 1).float()
        return (pos_mask * SurrogateMetrics.soft_indicator_lt(scores, threshold, temperature)).sum()
    
    @staticmethod
    def soft_precision(scores, labels, threshold, temperature=1.0):
        """Surrogate precision: TP / (TP + FP)"""
        tp = SurrogateMetrics.soft_tp(scores, labels, threshold, temperature)
        fp = SurrogateMetrics.soft_fp(scores, labels, threshold, temperature)
        return tp / (tp + fp + 1e-8)
    
    @staticmethod
    def soft_recall(scores, labels, threshold, temperature=1.0):
        """Surrogate recall: TP / (TP + FN)"""
        tp = SurrogateMetrics.soft_tp(scores, labels, threshold, temperature)
        fn = SurrogateMetrics.soft_fn(scores, labels, threshold, temperature)
        return tp / (tp + fn + 1e-8)


class ICO_PRAUC_Loss(nn.Module):
    """
    Implicit Constrained Optimization Loss для PR-AUC.
    
    PR-AUC аппроксимируется Riemann-суммой:
      PR-AUC ≈ (1/m) Σ precision(s_θ, λ_i)
    при ограничениях:
      recall(s_θ, λ_i) = β_i, для β_i = i/m, i=1,...,m
    
    Порог λ_i — неявная функция параметров θ:
      dλ/dθ = -∇_θ g̃(θ,λ) / (∂g̃/∂λ)  (через IFT)
    
    Итоговый градиент:
      ∇_θ f̃(θ, h(θ)) = ∇_θ f̃ + Σ_i (∂f̃/∂λ_i) * H_i
    где H_i = -∇_θ g̃_i / (∂g̃_i/∂λ_i)
    """
    
    def __init__(self, num_thresholds=20, temperature=1.0, recall_range=(0.1, 1.0)):
        super().__init__()
        self.m = num_thresholds
        self.temperature = temperature
        
        # Target recall values: β_1, ..., β_m — равномерно в [recall_range]
        self.beta = torch.linspace(recall_range[0], recall_range[1], self.m)
        
        # Learnable thresholds λ_1, ..., λ_m (инициализируем равномерно)
        self.thresholds = nn.Parameter(torch.linspace(-2.0, 2.0, self.m))
    
    def correction_step(self, scores, labels):
        """
        Коррекция порогов: находим λ_i такие что recall(s_θ, λ_i) = β_i
        используя unrelaxed (hard) метрики. Line search по sorted scores.
        """
        with torch.no_grad():
            sorted_scores = torch.sort(scores[labels == 1], descending=True).values
            total_pos = (labels == 1).sum().float()
            
            for i, beta in enumerate(self.beta):
                target_tp = beta * total_pos
                idx = min(int(target_tp.item()), len(sorted_scores) - 1)
                if idx >= 0 and idx < len(sorted_scores):
                    self.thresholds.data[i] = sorted_scores[idx]
    
    def forward(self, scores, labels, apply_stop_gradient=True):
        """
        Вычисляет ICO loss для оптимизации PR-AUC.
        
        Использует формулу (6) из статьи Kumar et al.:
        ∇_θ f̃(θ, h(θ)) = ∇_θ f̃(θ, λ) - ∇_θ Σ_i r_i * g̃_i(θ, λ_i)
        где r_i = (∂f̃/∂λ_i) / (∂g̃_i/∂λ_i)
        """
        device = scores.device
        beta = self.beta.to(device)
        thresholds = self.thresholds.to(device)
        
        sm = SurrogateMetrics
        tau = self.temperature
        
        # f̃ = -(1/m) Σ precision(s_θ, λ_i)  (negative because we minimize)
        precisions = []
        constraints = []
        
        for i in range(self.m):
            prec_i = sm.soft_precision(scores, labels, thresholds[i], tau)
            precisions.append(prec_i)
            
            recall_i = sm.soft_recall(scores, labels, thresholds[i], tau)
            g_i = recall_i - beta[i]
            constraints.append(g_i)
        
        # Objective: f̃ = -(1/m) Σ prec_i
        f_tilde = -torch.stack(precisions).mean()
        
        if apply_stop_gradient:
            # Формула (6): вычисляем r_i = (∂f̃/∂λ_i) / (∂g̃_i/∂λ_i) как stop-gradient
            r_values = []
            for i in range(self.m):
                # ∂f̃/∂λ_i
                df_dlambda = torch.autograd.grad(
                    -precisions[i] / self.m, thresholds[i:i+1], 
                    retain_graph=True, create_graph=False
                )[0]
                
                # ∂g̃_i/∂λ_i
                dg_dlambda = torch.autograd.grad(
                    constraints[i], thresholds[i:i+1],
                    retain_graph=True, create_graph=False
                )[0]
                
                r_i = (df_dlambda / (dg_dlambda + 1e-8)).detach()
                r_values.append(r_i)
            
            # Loss = f̃ - Σ r_i * g̃_i
            constraint_correction = sum(
                r_values[i] * constraints[i] for i in range(self.m)
            )
            loss = f_tilde - constraint_correction
        else:
            # Simplified version: just optimize f̃
            loss = f_tilde
        
        return loss, {
            'prauc_approx': -f_tilde.item(),
            'mean_precision': torch.stack(precisions).mean().item(),
            'constraint_violation': torch.stack([c.abs() for c in constraints]).mean().item(),
        }


class FocalLoss(nn.Module):
    """Focal Loss для сильного дисбаланса классов."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Комбинация: Focal Loss (для стабильной базовой оптимизации) 
    + ICO PR-AUC Loss (для прямой оптимизации целевой метрики).
    
    Schedule: начинаем с Focal, постепенно добавляем ICO.
    """
    
    def __init__(self, num_thresholds=20, temperature=1.0, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 ico_weight_start=0.0, ico_weight_end=1.0, warmup_epochs=5):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.ico = ICO_PRAUC_Loss(num_thresholds, temperature)
        self.ico_weight_start = ico_weight_start
        self.ico_weight_end = ico_weight_end
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def forward(self, logits, targets):
        # Focal component
        focal_loss = self.focal(logits, targets)
        
        # ICO component (using sigmoid of logits as scores)
        scores = torch.sigmoid(logits)
        ico_loss, ico_info = self.ico(scores, targets)
        
        # Weight schedule: linear warmup
        if self.current_epoch < self.warmup_epochs:
            w = self.ico_weight_start + (self.ico_weight_end - self.ico_weight_start) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            w = self.ico_weight_end
        
        total_loss = (1 - w) * focal_loss + w * ico_loss
        
        info = {
            'focal_loss': focal_loss.item(),
            'ico_loss': ico_loss.item(),
            'ico_weight': w,
            **ico_info,
        }
        
        return total_loss, info


# ============================================================
# Custom LightGBM objective для PR-AUC proxy
# ============================================================

def prauc_proxy_objective(y_pred, dtrain):
    """
    Custom objective для LightGBM, приближающая PR-AUC.
    
    Идея: вместо стандартной logloss, добавляем term который 
    штрафует модель за неправильный ранжирование positives vs negatives
    с учётом precision-recall tradeoff.
    
    Используем взвешенный logloss где веса зависят от текущего предсказания:
    - Для positives: больший вес для FN с высоким порогом (high recall region)
    - Для negatives: больший вес для FP с низким порогом (high precision region)
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Базовый градиент (logloss)
    grad = p - y_true
    hess = p * (1 - p)
    
    # Дополнительные веса для PR-AUC
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    # Для PR-AUC: важнее правильно ранжировать positives выше negatives
    # Усиливаем градиент для false negatives с высоким скором
    # и false positives с низким скором
    
    # Weight positives more when they have low predictions (need to push up)
    pos_weight = np.where(pos_mask, (1 - p) ** 0.5, 1.0)
    # Weight negatives more when they have high predictions (need to push down)  
    neg_weight = np.where(neg_mask, p ** 0.5, 1.0)
    
    weight = pos_weight * neg_weight
    
    # Scale by class imbalance
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    class_weight = np.where(pos_mask, n_neg / (n_pos + 1e-8), 1.0)
    weight *= class_weight
    
    grad *= weight
    hess *= weight
    
    return grad, hess


def prauc_eval_metric(y_pred, dtrain):
    """Custom eval metric для LightGBM: PR-AUC."""
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    score = average_precision_score(y_true, p)
    # LightGBM: (name, value, is_higher_better)
    return 'pr_auc', score, True


from sklearn.metrics import average_precision_score
