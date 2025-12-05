import math
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def mape_safe(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Calcula o MAPE ignorando casos com valor real zero para evitar divisao por zero."""
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_np) & np.isfinite(y_pred_np) & (y_true_np != 0)
    if not mask.any():
        return float("nan")
    ape = np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])
    return float(np.mean(ape))


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-6) -> float:
    """Calcula o SMAPE protegendo contra denominador zero."""
    y_true_np = np.asarray(y_true, float)
    y_pred_np = np.asarray(y_pred, float)
    denom = (np.abs(y_true_np) + np.abs(y_pred_np)).clip(min=eps)
    return float(np.mean(np.abs(y_pred_np - y_true_np) / denom))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Error com protecao de tipos."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error (raiz do MSE)."""
    mse_val = mean_squared_error(y_true, y_pred)
    return float(math.sqrt(mse_val)) if math.isfinite(mse_val) else float("nan")


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Coeficiente de determinacao R2."""
    return float(r2_score(y_true, y_pred))


def precision_from_mape(mape_val: float) -> float:
    """Converte MAPE em precisao (1 - MAPE)."""
    return 1.0 - mape_val


def _format_interval_bounds(lo_val: float, hi_val: float, mid_val: float) -> Tuple[float, float, float]:
    """Formata limites garantindo hi > lo e evita valores colados."""
    lo_disp = float(lo_val)
    hi_disp = float(hi_val if hi_val > lo_val else lo_val)
    if math.isclose(hi_disp, lo_disp, abs_tol=1e-3):
        hi_disp = lo_disp + 0.01
    mid_disp = float(mid_val)
    return lo_disp, hi_disp, mid_disp


def intervalo_90_catboost(y_low: float, y_mid: float, y_high: float) -> Dict[str, float]:
    """Monta dicionario de intervalo 90% (quantis) a partir dos quantis do CatBoost."""
    low_val = float(min(y_low, y_high))
    high_val = float(max(y_low, y_high))
    mid_val = float(y_mid)
    lo_disp, hi_disp, mid_disp = _format_interval_bounds(low_val, high_val, mid_val)
    return {
        "pred_mean": mid_val,
        "ci_low": low_val,
        "ci_high": high_val,
        "ci_low_disp": lo_disp,
        "ci_high_disp": hi_disp,
        "ci_mid_disp": mid_disp,
    }


def intervalo_90_bootstrap(preds: Iterable[float], q: Tuple[float, float] = (5, 95)) -> Dict[str, float]:
    """Gera intervalo 90% via bootstrap dos valores previstos."""
    preds_np = np.asarray(list(preds), dtype=float)
    if preds_np.size == 0:
        return {}
    lo_raw, hi_raw = np.percentile(preds_np, list(q))
    mean_val = float(np.mean(preds_np))
    lo_disp, hi_disp, mid_disp = _format_interval_bounds(lo_raw, hi_raw, mean_val)
    return {
        "pred_mean": mean_val,
        "ci_low": float(lo_raw),
        "ci_high": float(hi_raw),
        "ci_low_disp": float(lo_disp),
        "ci_high_disp": float(hi_disp),
        "ci_mid_disp": float(mid_disp),
    }
