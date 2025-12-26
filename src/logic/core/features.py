#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Imports
# =============================================================================
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..models.model_fila import DEFAULT_OCUPACAO_ALVO


# =============================================================================
# Helpers internos
# =============================================================================
def _ensure_series(val, idx) -> pd.Series:
    if isinstance(val, pd.Series):
        return val
    return pd.Series(val, index=idx)


# =============================================================================
# Public API
# =============================================================================
def criar_features_fila(df: pd.DataFrame, rho_target: float = DEFAULT_OCUPACAO_ALVO) -> pd.DataFrame:
    """Adiciona c_fila_continuo e auxiliares derivados via teoria das filas."""
    if df is None or df.empty:
        return df
    lambda_series = pd.to_numeric(df.get("demanda_hora"), errors="coerce").fillna(0.0)
    tma_candidates = [
        "tma_min",
        "TMA_min",
        "TMA",
        "TempoMedio",
        "TempoMedioAtendimento",
        "Tempo Medio Atendimento",
    ]
    tma_series = None
    for col in tma_candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                tma_series = series
                break
    if tma_series is None:
        tma_series = pd.Series(6.0, index=df.index, dtype="float64")
    tma_series = tma_series.clip(lower=1e-3)
    mu_series = 60.0 / tma_series
    denom = (rho_target * mu_series).replace(0, np.nan)
    c_fila = (lambda_series / denom).replace([np.inf, -np.inf], np.nan)
    c_fila = c_fila.fillna(1.0)
    c_fila = c_fila.clip(lower=1.0)
    rho_real = (lambda_series / (c_fila * mu_series)).replace([np.inf, -np.inf], np.nan)
    df["c_fila_continuo"] = c_fila
    df["rho_fila_continuo"] = rho_real
    return df


def criar_features_operacionais(
    df: pd.DataFrame,
    *,
    fit_cluster: bool = False,
    cluster_model: Optional["KMeans"] = None,
) -> Tuple[pd.DataFrame, Optional["KMeans"]]:
    """Enriquece o dataframe com features operacionais e fila continua (sem clusters)."""
    if df is None or df.empty:
        return df, cluster_model
    work = df.copy()
    idx = work.index
    dias = pd.to_numeric(work.get("DiasOperacionais"), errors="coerce").fillna(6.0)
    dias = dias.clip(lower=1.0)
    work["DiasOperacionais"] = dias
    horas = pd.to_numeric(work.get("HorasOperacionais"), errors="coerce").fillna(8.0)
    horas = horas.clip(lower=1.0)
    work["HorasOperacionais"] = horas

    volume_candidates = [
        "VolumeTotal",
        "BaseTotal",
        "BaseAtiva",
        "Pedidos/Dia",
        "Pedidos/Hora",
    ]
    volume = pd.Series(0.0, index=idx, dtype="float64")
    for col in volume_candidates:
        if col not in work.columns:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        if col == "Pedidos/Dia":
            candidate = series * dias
        elif col == "Pedidos/Hora":
            candidate = series * dias * horas
        else:
            candidate = series
        if candidate.notna().any():
            volume = candidate.fillna(volume)
            break
    denom_dia = dias.replace(0, np.nan)
    denom_hora = (dias * horas).replace(0, np.nan)
    demanda_dia = (volume / denom_dia).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    demanda_hora = (volume / denom_hora).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["demanda_dia"] = demanda_dia
    work["demanda_hora"] = demanda_hora

    base_ativa = _ensure_series(pd.to_numeric(work.get("BaseAtiva"), errors="coerce"), idx)
    work["base_ativa_dia"] = (base_ativa / denom_dia).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    receita_total = _ensure_series(pd.to_numeric(work.get("ReceitaTotalMes"), errors="coerce"), idx)
    work["receita_dia"] = (receita_total / denom_dia).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["receita_hora"] = (receita_total / denom_hora).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    qtd_caixas = _ensure_series(pd.to_numeric(work.get("Qtd Caixas"), errors="coerce").fillna(0.0), idx)
    work["capacidade_caixa_teorica"] = (qtd_caixas * horas).fillna(0.0)
    work["receita_por_caixa"] = (
        receita_total / qtd_caixas.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    reais_por_ativo = _ensure_series(pd.to_numeric(work.get("ReaisPorAtivo"), errors="coerce"), idx)
    work["reais_por_ativo_dia"] = (
        reais_por_ativo / denom_dia
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    i_cols = [c for c in ["I4", "I5", "I6", "I4aI6"] if c in work.columns]
    if i_cols:
        i_total = work[i_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    else:
        i_total = pd.Series(0.0, index=idx, dtype="float64")
    work["i4i6_total"] = i_total.fillna(0.0)
    work["i4i6_por_dia"] = (
        work["i4i6_total"] / denom_dia
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work["i4i6_por_hora"] = (
        work["i4i6_total"] / denom_hora
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    work = criar_features_fila(work, rho_target=DEFAULT_OCUPACAO_ALVO)
    work["cluster_loja"] = 0
    return work, None
