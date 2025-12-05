import math
from typing import Dict

from ..utils.helpers import safe_float

# Parametros padrao da teoria das filas
DEFAULT_OCUPACAO_ALVO = 0.80        # % do tempo produtivo (0-1)
QUEUE_CALIBRATION_DEFAULT = 1.0     # fator base (fixo) para calibrar a carga do modelo de filas


def _safe_float(val, default: float = 0.0) -> float:
    """Conversao defensiva para float reutilizando o helper compartilhado."""
    return safe_float(val, default)


def _estimate_operating_hours_per_day(feature_row: Dict[str, object]) -> float:
    """Retorna horas operacionais por dia da loja (em horas)."""
    horas_dia = _safe_float(feature_row.get("HorasOperacionais"), 0.0)
    dias_oper = _safe_float(feature_row.get("DiasOperacionais"), 0.0) or 6.0
    if horas_dia <= 0:
        horas_form = _safe_float(feature_row.get("horas_operacionais_form"), 0.0)
        if horas_form > 0:
            horas_dia = horas_form / max(dias_oper, 1.0) if horas_form > 24 else horas_form
    if horas_dia <= 0:
        horas_semana = _safe_float(feature_row.get("HorasOperacionaisSemana"), 0.0)
        if horas_semana > 0:
            horas_dia = horas_semana / max(dias_oper, 1.0)
    if horas_dia <= 0:
        horas_dia = _safe_float(feature_row.get("horas_loja_config"), 0.0)
        if horas_dia > 24:
            horas_dia = horas_dia / max(dias_oper, 1.0)
    return max(horas_dia, 1.0)


def _estimate_arrival_rate(feature_row: Dict[str, object]) -> float:
    """Estimativa de lambda (atendimentos/hora)."""
    horas_dia = _estimate_operating_hours_per_day(feature_row)
    pedidos_hora = _safe_float(feature_row.get("Pedidos/Hora"), 0.0)
    if pedidos_hora <= 0:
        pedidos_dia = _safe_float(feature_row.get("Pedidos/Dia"), 0.0)
        if pedidos_dia > 0 and horas_dia > 0:
            pedidos_hora = pedidos_dia / max(horas_dia, 1e-3)
    return max(pedidos_hora, 0.0)


def _estimate_service_time_minutes(feature_row: Dict[str, object]) -> float:
    """Tenta inferir o tempo medio de atendimento (MINUTOS)."""
    candidate_keys = [
        "TempoMedioAtendimento",
        "Tempo Medio Atendimento",
        "TempoMedio",
        "Tempo Medio",
        "tmedio_min_atendimento",
        "TMA",
        "TMA_min",
    ]
    for key in candidate_keys:
        if key in feature_row:
            val = _safe_float(feature_row.get(key), 0.0)
            if val > 0:
                if val > 300:  # valores muito altos provavelmente estao em segundos
                    val = val / 60.0
                elif val < 0.5:  # valores muito baixos provavelmente estao em horas
                    val = val * 60.0
                return val
    return 6.0


def estimate_queue_inputs(feature_row: Dict[str, object]) -> Dict[str, float]:
    """
    Retorna parametros (, TMA e ) usados pelos diagnosticos de fila.

     e sempre clientes por hora (convertemos volumes diarios dividindo pelas horas operacionais);
    TMA retorna minutos por atendimento;  = 1/TMA em horas.
    """
    lambda_hora = _estimate_arrival_rate(feature_row)
    tma_min = _estimate_service_time_minutes(feature_row)
    tma_hora = tma_min / 60.0
    mu_hora = 0.0 if tma_hora <= 0 else 1.0 / tma_hora
    return {
        "lambda_hora": float(lambda_hora),
        "tma_min": float(tma_min),
        "tma_hora": float(tma_hora),
        "mu_hora": float(mu_hora),
    }


def diagnosticar_fila(lambda_hora: float, tma_min: float, capacidade: float) -> Dict[str, float]:
    """
    Calcula metricas basicas da fila M/M/c para uma capacidade fornecida (headcount).

    Parametros:
        lambda_hora : chegadas por HORA (atendimentos/h)
        tma_min     : tempo medio de atendimento em MINUTOS
        capacidade  : numero de auxiliares (float)
    """
    lambda_hora = max(float(lambda_hora), 0.0)
    tma_min = max(float(tma_min), 1e-6)
    tma_hora = tma_min / 60.0  # horas por atendimento
    mu_hora = 0.0 if tma_hora <= 0 else 1.0 / tma_hora
    capacidade = max(float(capacidade), 0.0)
    if capacidade <= 0 or mu_hora <= 0:
        rho = float("nan")
    else:
        rho = lambda_hora / (capacidade * mu_hora)
    return {
        "lambda_hora": lambda_hora,
        "tma_min": tma_min,
        "mu_hora": mu_hora,
        "capacidade": capacidade,
        "rho": rho,
    }


def calcular_fila(
    lambda_hora: float,
    tma_min: float,
    rho_target: float = DEFAULT_OCUPACAO_ALVO,
    calibration_factor: float = QUEUE_CALIBRATION_DEFAULT,
) -> Dict[str, float]:
    """
    Calcula o headcount bruto e a utilizacao alvo segundo a teoria das filas.

    A calibragem (facultativa) apenas escala a carga efetiva .
    """
    lambda_hora = max(float(lambda_hora), 0.0)
    tma_min = max(float(tma_min), 1e-6)
    if not math.isfinite(rho_target) or rho_target <= 0:
        rho_target = DEFAULT_OCUPACAO_ALVO
    rho_target = float(min(0.99, max(rho_target, 1e-3)))
    tma_hora = tma_min / 60.0
    mu_hora = 0.0 if tma_hora <= 0 else 1.0 / tma_hora
    if not math.isfinite(calibration_factor) or calibration_factor <= 0:
        calibration_factor = QUEUE_CALIBRATION_DEFAULT
    lambda_efetivo = lambda_hora * calibration_factor
    if mu_hora <= 0 or rho_target <= 0:
        c_fila_bruto = float("nan")
        c_fila = 1.0
    else:
        c_fila_bruto = lambda_efetivo / (rho_target * mu_hora)
        c_fila = max(1.0, float(c_fila_bruto))
    denom = c_fila * mu_hora
    rho_fila = float("nan") if denom <= 0 else lambda_hora / denom
    return {
        "lambda_hora": float(lambda_hora),
        "lambda_efetivo": float(lambda_efetivo),
        "tma_min": float(tma_min),
        "tma_hora": float(tma_hora),
        "mu_hora": float(mu_hora),
        "rho_target": float(rho_target),
        "calibration_factor": float(calibration_factor),
        "c_fila_bruto": float(c_fila_bruto),
        "c_fila": float(c_fila),
        "rho_fila": float(rho_fila),
    }
