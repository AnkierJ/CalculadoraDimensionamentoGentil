#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from sklearn.cluster import KMeans
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
import os
import csv
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
except ImportError:
    variance_inflation_factor = None
from ..models.model_catboost import CatBoostQuantileModel, predict_catboost, train_catboost_model
from ..models.model_fila import (
    DEFAULT_OCUPACAO_ALVO,
    diagnosticar_fila,
    estimate_queue_inputs,
)
from ..utils.helpers import (
    FALSE_BOOL_VALUES,
    SYN,
    TRUE_BOOL_VALUES,
    _coerce_types,
    _ensure_columns,
    _norm_code,
    _standardize_cols,
    _standardize_row,
    calc_pct,
    create_empty_from_schema,
    get_lookup,
    get_lookup_value,
    get_schema_dAmostras,
    get_schema_dEstrutura,
    get_schema_dPessoas,
    get_schema_fFaturamento2,
    get_schema_fIndicadores,
    image_to_base64,
    normalize_processo_nome,
    read_csv_with_schema,
    safe_float,
    template_df,
    to_csv_bytes,
    validate_df,
)
from ..utils.metrics import (
    mae,
    mape_safe,
    precision_from_mape,
    r2,
    rmse,
    smape,
    intervalo_90_catboost,
)
from ..data.buscaDeLojas import (
    _ensure_loja_key,
    _get_loja_row,
    _filter_df_by_loja,
    carregar_lojas,
    listar_nomes_lojas,
    obter_loja_por_nome,
    filtrar_lojas,
)
from .features import criar_features_operacionais, criar_features_fila
from .clusters import treinar_kmeans, atribuir_cluster
# Parâmetros padrão para o modo IDEAL
DEFAULT_ABSENTEISMO   = 0.08        # férias + faltas + treinamentos (0–1)
DEFAULT_SLA_BUFFER    = 0.05        # folga extra p/ pico/SLA além da margem
WEEKS_PER_MONTH = 4.33
# =============================================================================
# Helpers importados de utils.helpers
# =============================================================================
# =============================================================================
# Processos e cargas
# =============================================================================
def agregar_tempo_medio_por_processo(amostras: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas: Loja, Processo, tempo_medio_min
    Usa 'Tempo Médio' se existir; caso contrário, usa média de 'Minutos'.
    """
    if amostras is None or amostras.empty:
        return pd.DataFrame(columns=["Loja", "Processo", "tempo_medio_min"])
    df = amostras.copy()
    has_tempo_medio = "Tempo Médio" in df.columns and df["Tempo Médio"].notna().any()
    if has_tempo_medio:
        grouped = df.groupby(["Loja", "Processo"], dropna=False)["Tempo Médio"].mean().reset_index()
        grouped = grouped.rename(columns={"Tempo Médio": "tempo_medio_min"})
    else:
        grouped = df.groupby(["Loja", "Processo"], dropna=False)["Minutos"].mean().reset_index()
        grouped = grouped.rename(columns={"Minutos": "tempo_medio_min"})
    return grouped
def inferir_frequencia_por_processo(amostras: pd.DataFrame) -> pd.DataFrame:
    """
    Deprecated: frequência não é inferida de dAmostras.
    Mantém compatibilidade retornando DataFrame vazio.
    """
    return pd.DataFrame(columns=["Loja", "Processo", "frequencia"])
def calcular_carga_por_processo(tempos_processo: pd.DataFrame, frequencias: pd.DataFrame, fator_monotonia: float = 1.0) -> Tuple[pd.DataFrame, float]:
    """
    Junta tempos por processo (Loja, Processo, tempo_medio_min) com frequências (Loja, Processo, frequencia)
    e calcula carga por processo em horas, retornando (detalhe_df, carga_total_horas)
    """
    if tempos_processo is None or tempos_processo.empty:
        return pd.DataFrame(columns=["Loja", "Processo", "tempo_medio_min", "frequencia", "carga_horas"]), 0.0
    if frequencias is None or "frequencia" not in frequencias.columns:
        # assume zero se não fornecido
        frequencias = tempos_processo[["Loja", "Processo"]].copy()
        frequencias["frequencia"] = 0
    base = pd.merge(tempos_processo, frequencias, on=["Loja", "Processo"], how="left")
    base["frequencia"] = pd.to_numeric(base["frequencia"], errors="coerce").fillna(0)
    base["tempo_medio_min"] = pd.to_numeric(base["tempo_medio_min"], errors="coerce").fillna(0)
    base["carga_horas"] = base["frequencia"] * base["tempo_medio_min"] / 60.0
    carga_total = float(base["carga_horas"].sum()) * fator_monotonia
    return base, carga_total
def estimate_fluxo_medio_indicadores(
    base_ativa: float,
    receita_total_mes: float,
    pedidos_dia_hist: float,
    pedidos_hora_hist: float,
    dias_operacionais: float,
    horas_operacionais_semanais: float,
) -> Dict[str, float]:
    """
    Combina indicadores comerciais para estimar pedidos por semana/hora.
    Retorna dict com pedidos_semana, pedidos_hora e componentes usados.
    """
    def _safe_value(val: Optional[float]) -> float:
        try:
            v = float(val)
            if not math.isfinite(v):
                return 0.0
            return max(0.0, v)
        except Exception:
            return 0.0
    dias = _safe_value(dias_operacionais)
    if dias <= 0:
        dias = 6.0
    dias = max(1.0, min(7.0, dias))
    horas_semanais = _safe_value(horas_operacionais_semanais)
    if horas_semanais <= 0:
        horas_semanais = dias * 10.0
    horas_semanais = max(dias, horas_semanais)
    base_ativa_v = _safe_value(base_ativa)
    receita_mes = _safe_value(receita_total_mes)
    pedidos_dia_hist = _safe_value(pedidos_dia_hist)
    pedidos_hora_hist = _safe_value(pedidos_hora_hist)
    componentes: Dict[str, float] = {}
    candidatos: List[Tuple[str, float, float]] = []
    if receita_mes > 0 and base_ativa_v > 0:
        pedidos_semana_rec = (receita_mes / max(base_ativa_v, 1e-6)) / WEEKS_PER_MONTH
        componentes["receita_base"] = pedidos_semana_rec
        candidatos.append(("receita_base", pedidos_semana_rec, 1.0))
    if pedidos_dia_hist > 0:
        pedidos_semana_hist = pedidos_dia_hist * dias
        componentes["pedidos_dia"] = pedidos_semana_hist
        candidatos.append(("pedidos_dia", pedidos_semana_hist, 0.8))
    if pedidos_hora_hist > 0:
        pedidos_semana_hora = pedidos_hora_hist * horas_semanais
        componentes["pedidos_hora"] = pedidos_semana_hora
        candidatos.append(("pedidos_hora", pedidos_semana_hora, 0.9))
    pedidos_semana = 0.0
    if candidatos:
        soma_pesos = sum(peso for _, _, peso in candidatos)
        pedidos_semana = sum(valor * peso for _, valor, peso in candidatos) / max(soma_pesos, 1e-6)
    pedidos_hora = pedidos_semana / max(horas_semanais, 1.0)
    return {
        "pedidos_semana": float(pedidos_semana),
        "pedidos_hora": float(pedidos_hora),
        "componentes": componentes,
    }
def estimate_pedidos_por_hora(
    indicadores: Dict[str, float],
    horas_operacionais_semanais: float,
    dias_operacionais_semana: float,
) -> float:
    """
    Estima pedidos/hora usando indicadores diretos (Pedidos/Hora, Pedidos/Dia, Faturamento).
    dias_operacionais_semana é usado para converter pedidos/dia em semanais.
    """
    base = safe_float(indicadores.get("Pedidos/Hora"), 0.0)
    horas_semanais = max(1.0, safe_float(horas_operacionais_semanais, 0.0))
    dias = safe_float(dias_operacionais_semana, 0.0)
    if dias <= 0:
        dias = 6.0
    dias = max(1.0, min(7.0, dias))
    if base <= 0 and horas_semanais > 0:
        pedidos_dia = safe_float(indicadores.get("Pedidos/Dia"), 0.0)
        if pedidos_dia > 0:
            pedidos_semana = pedidos_dia * dias
            base = pedidos_semana / horas_semanais
    return max(base, 0.0)
def estimate_process_frequencies_from_indicadores(
    base_ativa: float,
    recuperados: float,
    inicios: float,
    reinicios: float,
    pedidos_semana: float,
    itens_por_pedido: float,
    pct_retirada: float,
    dias_operacionais: float,
) -> Dict[str, float]:
    """
    Estima frequências semanais para os processos prioritários a partir dos indicadores inseridos.
    """
    def _safe_positive(val: Optional[float]) -> float:
        try:
            v = float(val)
            if not math.isfinite(v):
                return 0.0
            return max(0.0, v)
        except Exception:
            return 0.0
    dias = _safe_positive(dias_operacionais)
    if dias <= 0:
        dias = 6.0
    dias = max(1.0, min(7.0, dias))
    pedidos_semana = _safe_positive(pedidos_semana)
    if pedidos_semana <= 0:
        pedidos_semana = 0.0
    itens_por_pedido = max(0.1, _safe_positive(itens_por_pedido))
    pct_retirada = max(0.0, min(100.0, _safe_positive(pct_retirada)))
    base_ativa_v = _safe_positive(base_ativa)
    recuperados_v = _safe_positive(recuperados)
    inicios_v = _safe_positive(inicios)
    reinicios_v = _safe_positive(reinicios)
    resultado: Dict[str, float] = {}
    resultado_key = normalize_processo_nome("Reposição de prateleira (estoque)")
    resultado[resultado_key] = (pedidos_semana * itens_por_pedido) / 30.0
    resultado_key = normalize_processo_nome("Separação de mercadoria (on-line e retirada)")
    resultado[resultado_key] = pedidos_semana * (0.5 + 0.5 * (pct_retirada / 100.0))
    resultado_key = normalize_processo_nome("Faturamente de pedido (retirada e delivery)")
    resultado[resultado_key] = pedidos_semana
    resultado_key = normalize_processo_nome("Devolução")
    churn_component = (base_ativa_v) / WEEKS_PER_MONTH
    resultado[resultado_key] = (recuperados_v * 0.5) + churn_component
    resultado_key = normalize_processo_nome("Cadastro de revendedor")
    resultado[resultado_key] = inicios_v / WEEKS_PER_MONTH
    resultado_key = normalize_processo_nome("Atualização de cadastro de revendedor")
    resultado[resultado_key] = reinicios_v / WEEKS_PER_MONTH
    resultado_key = normalize_processo_nome("Abertura e acompanhamento de chamado")
    resultado[resultado_key] = (recuperados_v * 0.8) / max(dias, 1.0)
    resultado_key = normalize_processo_nome("Eventos para os revendedores")
    resultado[resultado_key] = 0.0
    return {k: float(max(0.0, v)) for k, v in resultado.items()}
# =============================================================================
# Preparação de features e modelagem
# =============================================================================
def _bool_to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Converte campos booleanos em inteiros 0/1 para alimentar o modelo."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            series = df[c]
            if pd.api.types.is_bool_dtype(series):
                bool_ser = series.fillna(False)
            elif pd.api.types.is_numeric_dtype(series):
                bool_ser = series.fillna(0).astype(float).ne(0)
            else:
                s = series.astype(str).str.strip().str.upper()
                bool_ser = pd.Series(pd.NA, index=s.index, dtype="boolean")
                bool_ser = bool_ser.mask(s.isin(TRUE_BOOL_VALUES), True)
                bool_ser = bool_ser.mask(s.isin(FALSE_BOOL_VALUES), False)
                bool_ser = bool_ser.fillna(False)
            df[c] = bool_ser.astype(int)
    return df
def prepare_training_dataframe(dEstrutura, dPessoas, fIndicadores) -> pd.DataFrame:
    """Combina estrutura, pessoas e indicadores para montar o dataset de treino."""
    if dEstrutura is None or dEstrutura.empty: return pd.DataFrame()
    if dPessoas is None or dPessoas.empty: return pd.DataFrame()
    dEstrutura = _ensure_loja_key(dEstrutura)
    dPessoas = _ensure_loja_key(dPessoas)
    fIndicadores = _ensure_loja_key(fIndicadores)
    df = dEstrutura.copy()
    key_col = "Loja_norm" if "Loja_norm" in df.columns else "Loja"
    if "DiasOperacionais" in df.columns:
        df["DiasOperacionais"] = pd.to_numeric(df["DiasOperacionais"], errors="coerce").clip(1, 7)
        df["DiasOperacionais"] = df["DiasOperacionais"].fillna(6.0)
    else:
        df["DiasOperacionais"] = 6.0
    # normalizações de nomes
    if "Qtd Caixas" not in df.columns and "Caixas" in df.columns:
        df["Qtd Caixas"] = df["Caixas"]
    if "Espaco Evento" not in df.columns and "Esp Conv" in df.columns:
        df["Espaco Evento"] = df["Esp Conv"]
    # Horas operacionais (se vierem strings "HH:MM")
    for c in ["HoraAbertura","HoraFechamento"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.time
    if {"HoraAbertura","HoraFechamento"}.issubset(df.columns):
        def _horas_op(row):
            try:
                a = pd.Timestamp.combine(pd.Timestamp.today().date(), row["HoraAbertura"])
                f = pd.Timestamp.combine(pd.Timestamp.today().date(), row["HoraFechamento"])
                h = (f - a).total_seconds()/3600.0
                return max(0.0, h if h <= 24 else 0.0)
            except Exception:
                return pd.NA
        df["HorasOperacionais"] = df.apply(_horas_op, axis=1)
    # target
    merge_key = key_col if key_col in dPessoas.columns else "Loja"
    df = pd.merge(df, dPessoas[[merge_key,"QtdAux"]], left_on=key_col, right_on=merge_key, how="inner")
    # indicadores (agregar por loja)
    if fIndicadores is not None and not fIndicadores.empty:
        ind_keep = [
            "Loja",
            "BaseAtiva",
            "%Ativos",
            "TaxaInicios",
            "TaxaReativacao",
            "Pedidos/Hora",
            "Pedidos/Dia",
            "Itens/Pedido",
            "Faturamento/Hora",
            "%Retirada",
            "ReceitaTotalMes",
        ]
        cols = [c for c in ind_keep if c in fIndicadores.columns]
        group_col = key_col if key_col in fIndicadores.columns else "Loja"
        ind = fIndicadores.groupby(group_col, as_index=False)[cols[1:]].mean() if len(cols)>1 else None
        if ind is not None:
            df = pd.merge(df, ind, left_on=key_col, right_on=group_col, how="left")
    # booleans→int
    df = _bool_to_int(df, ["Escritorio","Copa","Espaco Evento"])
    # numericos
    for c in set(FEATURE_COLUMNS + ["QtdAux"]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
    extra_numeric = [
        "BaseTotal",
        "BaseAtiva",
        "ReceitaTotalMes",
        "Inicios",
        "Reinicios",
        "Recuperados",
        "I4aI6",
        "AtividadeER",
        "ReceitaPorAux",
    ]
    for extra_col in extra_numeric:
        if extra_col in df.columns:
            df[extra_col] = pd.to_numeric(df[extra_col].astype(str).str.replace(",", "."), errors="coerce")
    if {"ReceitaTotalMes", "QtdAux"}.issubset(df.columns):
        receita = pd.to_numeric(df["ReceitaTotalMes"], errors="coerce")
        qtd_aux_hist = pd.to_numeric(df["QtdAux"], errors="coerce").replace(0, pd.NA)
        df["ReceitaPorAux"] = (receita / qtd_aux_hist).replace([float("inf"), float("-inf")], pd.NA)
    def _fill_ratio(target: str, numerator_col: str, denom_col: str, fallback_col: Optional[str] = None, multiplier: float = 1.0):
        if numerator_col not in df.columns:
            return
        if denom_col not in df.columns and (fallback_col is None or fallback_col not in df.columns):
            return
        denom = df[denom_col] if denom_col in df.columns else pd.Series(pd.NA, index=df.index)
        if fallback_col and fallback_col in df.columns:
            fallback = df[fallback_col]
            denom = denom.where((denom.notna()) & (denom > 0), fallback)
        denom = denom.replace(0, pd.NA)
        ratio = (df[numerator_col] / denom) * multiplier
        ratio = ratio.replace([float("inf"), float("-inf")], pd.NA)
        if target in df.columns:
            df[target] = df[target].fillna(ratio)
        else:
            df[target] = ratio
    # derivar se faltar
    if "%Ativos" not in df.columns:
        df["%Ativos"] = pd.NA
    _fill_ratio("%Ativos", "BaseAtiva", "BaseTotal", multiplier=100.0)
    if "TaxaInicios" not in df.columns:
        df["TaxaInicios"] = pd.NA
    _fill_ratio("TaxaInicios", "Inicios", "BaseAtiva", fallback_col="BaseTotal", multiplier=100.0)
    if "TaxaReativacao" not in df.columns:
        df["TaxaReativacao"] = pd.NA
    if {"Recuperados", "I4aI6"}.issubset(df.columns):
        denom = df["I4aI6"].replace(0, pd.NA)
        numer = df["Recuperados"]
        ratio = (numer / denom) * 100.0
        df["TaxaReativacao"] = df["TaxaReativacao"].fillna(ratio.replace([float("inf"), float("-inf")], pd.NA))
    else:
        _fill_ratio("TaxaReativacao", "Reinicios", "BaseAtiva", fallback_col="BaseTotal", multiplier=100.0)
    # limpa
    df = df.dropna(subset=["QtdAux"])
    # mantenha linhas com bastante feature; preencha poucos NaNs depois
    # se alguma feature existir mas está toda NaN, zere (evita median=NaN)
    for c in [col for col in FEATURE_COLUMNS if col in df.columns]:
        if df[c].isna().all():
            df[c] = 0.0
    df, cluster_model = criar_features_operacionais(df, fit_cluster=True)
    if cluster_model is not None:
        df.attrs["cluster_model"] = cluster_model
    if "Loja_norm" in df.columns:
        df = df.drop(columns=["Loja_norm"])
    return df
def _assign_clusters(
    df: pd.DataFrame,
    *,
    cluster_model: Optional["KMeans"],
    fit_new: bool = False,
) -> Tuple[pd.DataFrame, Optional["KMeans"]]:
    """Função mantida por compatibilidade; atualmente não aplica clustering."""
    if df is None or df.empty:
        return df, None
    df["cluster_loja"] = df.get("cluster_loja", 0).fillna(0).astype(int)
    return df, None
def _augment_feature_row(feature_row: Dict[str, object], cluster_model: Optional["KMeans"]) -> Dict[str, object]:
    """Compatibilidade: retorna a própria linha sem enriquecimento adicional."""
    return feature_row
def clean_training_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aplica limpezas adicionais (target válido, winsorização e drop de colunas fracas)."""
    if df is None or df.empty:
        return pd.DataFrame()
    cleaned = df.copy()
    if "Loja_norm" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Loja_norm"])
    if "QtdAux" in cleaned.columns:
        cleaned["QtdAux"] = pd.to_numeric(cleaned["QtdAux"], errors="coerce")
        cleaned = cleaned[(cleaned["QtdAux"].notna()) & (cleaned["QtdAux"] > 0)]
    if cleaned.empty:
        return cleaned
    cont_cols = [c for c in CONT if c in cleaned.columns]
    for c in cont_cols:
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce").astype(float)
        valid = cleaned[c].dropna()
        if len(valid) >= 5:
            lo, hi = valid.quantile([0.05, 0.95])
            if pd.notna(lo) and pd.notna(hi):
                cleaned[c] = cleaned[c].clip(lo, hi)
    drop_cols: List[str] = []
    for col in cleaned.columns:
        if col in ("Loja", "QtdAux"):
            continue
        series = cleaned[col]
        if series.notna().sum() < 3:
            drop_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            if float(series.std(skipna=True) or 0.0) == 0.0:
                drop_cols.append(col)
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    cleaned.attrs.update(getattr(df, "attrs", {}))
    return cleaned
FEATURE_COLUMNS = [
    # estrutura física
    "Area Total", "Qtd Caixas", "DiasOperacionais",
    "Escritorio", "Copa", "Espaco Evento",
    # demanda/fluxo
    "Pedidos/Hora", "Pedidos/Dia", "Itens/Pedido", "Faturamento/Hora", "%Retirada",
    # base comercial
    "BaseAtiva", "TaxaInicios", "TaxaReativacao",
    # disponibilidade/operacao
    "HorasOperacionais",
]
CONT = ["Area Total","Qtd Caixas","Pedidos/Hora","Pedidos/Dia",
        "Itens/Pedido","Faturamento/Hora","%Retirada","BaseAtiva",
        "TaxaInicios","TaxaReativacao","HorasOperacionais","DiasOperacionais"]
MODEL_ALGO_ORDER = ["catboost"]
MODEL_ALGO_NAMES = {
    "catboost": "CatBoostRegressor",
}


def _get_high_perf_anchor(train_df: pd.DataFrame, q: float = 0.75) -> Optional[float]:
    """
    Retorna um valor de referência (percentil superior) de Receita por Auxiliar.
    Usado para ancorar o alvo "ideal" nas lojas com melhor desempenho.
    """
    rec_por_aux = _compute_receita_por_aux(train_df, train_df.get("QtdAux"))
    rec_valid = rec_por_aux.dropna()
    rec_valid = rec_valid[rec_valid > 0]
    if rec_valid.empty:
        return None
    try:
        return float(rec_valid.quantile(q))
    except Exception:
        return None


def _apply_high_perf_cap(
    y_series: pd.Series,
    train_df: pd.DataFrame,
    margem: float,
    anchor_quantile: Optional[float] = None,
) -> pd.Series:
    """
    Limita o alvo ideal com base no desempenho das lojas de maior Receita/Aux.
    Evita que o headcount "ideal" extrapole o que seria necessário para atingir
    o nível de eficiência das top lojas.
    """
    q_cap = anchor_quantile if anchor_quantile is not None else 0.80
    anchor = _get_high_perf_anchor(train_df, q=q_cap)
    if anchor is None or anchor <= 0:
        return y_series
    receita = pd.to_numeric(train_df.get("ReceitaTotalMes"), errors="coerce")
    if receita is None or receita.empty:
        return y_series
    cap = (receita / anchor) * (1.0 + margem)
    cap = cap.replace([np.inf, -np.inf], np.nan)
    capped = y_series.copy()
    cap_aligned = cap.reindex(y_series.index)
    mask = cap_aligned.notna() & (cap_aligned > 0)
    capped.loc[mask] = np.minimum(capped.loc[mask], cap_aligned.loc[mask])
    return capped


def _feature_bounds_from_train(
    train_df: pd.DataFrame,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Calcula faixas de valores (quantis) das features numéricas para reuso na inferência.
    Evita extrapolar muito além do espaço visto no treino.
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    if train_df is None or train_df.empty:
        return bounds
    for col in [c for c in FEATURE_COLUMNS if c in train_df.columns]:
        series = pd.to_numeric(train_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        # Afrouxa o clipping para BaseAtiva (usa quantis 1% e 99%)
        lo_q, hi_q = (0.01, 0.99) if col == "BaseAtiva" else (lower_q, upper_q)
        lo, hi = series.quantile([lo_q, hi_q])
        if pd.notna(lo) and pd.notna(hi):
            bounds[col] = (float(lo), float(hi))
    return bounds


def _cap_feature_row(
    feature_row: Dict[str, object],
    bounds: Dict[str, Tuple[float, float]],
    skip_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Aplica clipping nos valores de entrada conforme os quantis do treino."""
    if not bounds or not feature_row:
        return feature_row
    capped = dict(feature_row)
    skip_set = set(skip_cols or [])
    for col, (lo, hi) in bounds.items():
        if col in skip_set or col not in capped:
            continue
        try:
            val = float(capped[col])
        except Exception:
            continue
        capped[col] = min(max(val, lo), hi)
    return capped

def _prepare_model_data(
    train_df: pd.DataFrame,
    mode: str,
    horas_disp: float,
    margem: float,
    anchor_quantile: Optional[float] = None,
) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]]:
    """
    Prepara matriz de atributos (X) e target (y) para uso pelos modelos de regressão.
    """
    if train_df is None or train_df.empty:
        return None
    train_df = clean_training_dataframe(train_df)
    if train_df is None or train_df.empty:
        return None
    used_features = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    if not used_features:
        return None
    preserve_cols = ["BaseAtiva", "Pedidos/Dia", "Pedidos/Hora", "Faturamento/Hora"]
    used_features = drop_high_correlation(
        train_df,
        used_features,
        thr=0.85,
        preserve_cols=preserve_cols,
    )
    X = train_df[used_features].copy()
    y = make_target(
        train_df,
        mode=mode,
        horas_disp=horas_disp,
        margem=margem,
        anchor_quantile=anchor_quantile,
    )
    y = pd.to_numeric(y, errors="coerce")
    mask_valid = y.notna()
    X, y = X.loc[mask_valid], y.loc[mask_valid]
    if X.empty:
        return None
    X, used_features = _reduce_features_by_mi(X, y, used_features)
    numeric_cols, categorical_cols = _infer_feature_types(X, used_features)
    X.attrs["numeric_features"] = numeric_cols
    X.attrs["categorical_features"] = categorical_cols
    base_weights = pd.Series(1.0, index=X.index, dtype="float64")
    sample_weights = base_weights.copy()
    if mode == "ideal":
        qtd_aux_hist = pd.to_numeric(train_df.get("QtdAux"), errors="coerce")
        receita_por_aux = _compute_receita_por_aux(train_df, qtd_aux_hist)
        receita_por_aux = receita_por_aux.loc[mask_valid]
        positivos = receita_por_aux[(receita_por_aux > 0) & receita_por_aux.notna()]
        if not positivos.empty:
            ref = positivos.median()
            if pd.isna(ref) or ref <= 0:
                ref = positivos.mean() or 1.0
            fator = (receita_por_aux / max(ref, 1e-6)).clip(lower=0.4, upper=2.5).fillna(1.0)
            sample_weights = fator.reindex(base_weights.index, fill_value=1.0)
    return X, y, used_features, sample_weights
def drop_high_correlation(
    df: pd.DataFrame,
    cols: List[str],
    thr: float = 0.85,
    preserve_cols: Optional[List[str]] = None,
) -> List[str]:
    """Remove features altamente correlacionadas para evitar multicolinearidade."""
    X = df[[c for c in cols if c in df.columns]].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(how="all", axis=1)
    if X.shape[1] <= 1:
        return list(X.columns)
    preserve = set(preserve_cols or [])
    # ordem por variância (labels, não posicional)
    var_order = X.var(numeric_only=True).sort_values(ascending=False)
    order = var_order.index.tolist()
    corr = X.corr(numeric_only=True).abs()
    to_drop: set = set()
    for i, c1 in enumerate(order):
        if c1 in to_drop:
            continue
        for c2 in order[i+1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] >= thr:
                # derruba a "pior": maior taxa de nulos; em empate, menor variância
                null1 = X[c1].isna().mean()
                null2 = X[c2].isna().mean()
                if null1 > null2:
                    if c1 not in preserve:
                        to_drop.add(c1)
                elif null2 > null1:
                    if c2 not in preserve:
                        to_drop.add(c2)
                else:
                    drop_candidate = c1 if var_order[c1] < var_order[c2] else c2
                    if drop_candidate not in preserve:
                        to_drop.add(drop_candidate)
    kept = [c for c in order if c not in to_drop]
    return kept
def _reduce_features_by_mi(
    X: pd.DataFrame,
    y: pd.Series,
    used_features: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Subseleciona features via informação mútua para bases muito pequenas."""
    n_samples = len(X)
    n_features = len(used_features)
    if n_samples == 0 or n_features <= 1:
        return X, used_features
    if n_samples >= 20 or n_features <= 6:
        return X, used_features
    k = max(3, min(n_features, max(2, n_samples - 1)))
    try:
        X_filled = X.fillna(X.median(numeric_only=True))
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X_filled, y.loc[X_filled.index])
        mask = selector.get_support()
        kept = [feat for feat, keep in zip(used_features, mask) if keep]
        if kept:
            return X[kept].copy(), kept
    except Exception:
        pass
    return X, used_features
def _infer_feature_types(
    X: pd.DataFrame,
    used_features: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Identifica automaticamente colunas numéricas e categóricas.
    Colunas listadas em CONT são tratadas como numéricas mesmo se vierem como texto.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    cols = used_features or list(X.columns)
    for col in cols:
        if col not in X.columns:
            continue
        series = X[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            X[col] = pd.to_numeric(series, errors="coerce")
            numeric_cols.append(col)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        total = max(len(series), 1)
        valid_ratio = coerced.notna().sum() / total
        if col in CONT or valid_ratio >= 0.7:
            X[col] = coerced
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols
def train_auxiliares_model(
    train_df: pd.DataFrame,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algo: str = "catboost",
    prepared: Optional[Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]] = None,
    anchor_quantile: Optional[float] = None,
) -> Optional[object]:
    """Treina o modelo (CatBoost) usando pipelines padronizados."""
    if prepared is None:
        prepared = _prepare_model_data(train_df, mode, horas_disp, margem, anchor_quantile)
    if prepared is None:
        return None
    X_full, y_full, used_features, sample_weights = prepared
    X = X_full.copy()
    y = y_full.copy()
    if isinstance(sample_weights, pd.Series):
        sample_weights = sample_weights.reindex(X.index).fillna(1.0)
    else:
        sample_weights = pd.Series(1.0, index=X.index, dtype="float64")
    algo = (algo or "catboost").lower()
    numeric_cols = X.attrs.get("numeric_features")
    categorical_cols = X.attrs.get("categorical_features")
    if numeric_cols is None or categorical_cols is None:
        numeric_cols, categorical_cols = _infer_feature_types(X, used_features)
    cat_cardinality = getattr(X_full, "attrs", {}).get("categorical_cardinality", {})
    if algo == "catboost":
        return train_catboost_model(
            X=X,
            y=y,
            used_features=used_features,
            numeric_cols=numeric_cols or [],
            categorical_cols=categorical_cols or [],
            sample_weights=sample_weights,
        )
    raise ValueError(f"Algoritmo '{algo}' não suportado.")
def _determine_test_fraction(n_samples: int, desired: float) -> float:
    """Ajusta o percentual da base de teste garantindo pelo menos 1 observação."""
    if n_samples <= 1:
        return 0.5
    min_frac = 1.0 / max(n_samples, 1)
    frac = max(desired, min_frac)
    return float(min(0.4, max(frac, 0.2)))
def _split_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[Dict[str, object]]:
    """Centraliza a divisão treino/teste para todos os algoritmos."""
    n_samples = len(X)
    if n_samples < 3:
        return None
    frac = _determine_test_fraction(n_samples, test_size)
    arrays: List[object] = [X, y]
    has_sw = isinstance(sample_weights, pd.Series)
    if has_sw:
        arrays.append(sample_weights)
    split = train_test_split(
        *arrays,
        test_size=frac,
        shuffle=True,
        random_state=random_state,
    )
    if has_sw:
        X_train, X_test, y_train, y_test, sw_train, sw_test = split
    else:
        X_train, X_test, y_train, y_test = split
        sw_train = sw_test = None
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sample_weight_train": sw_train,
        "sample_weight_test": sw_test,
    }
def train_all_auxiliares_models(
    train_df: pd.DataFrame,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algos: Optional[List[str]] = None,
    anchor_quantile: Optional[float] = None,
) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Treina todos os modelos disponíveis e retorna (modelos, erros)."""
    algos = algos or MODEL_ALGO_ORDER
    models: Dict[str, object] = {}
    errors: Dict[str, str] = {}
    prepared = _prepare_model_data(train_df, mode, horas_disp, margem, anchor_quantile)
    if prepared is None:
        errors["_geral"] = "Sem dados suficientes para treinar (faltam features ou target)."
        return models, errors
    for algo in algos:
        try:
            model = train_auxiliares_model(
                train_df,
                mode=mode,
                horas_disp=horas_disp,
                margem=margem,
                algo=algo,
                prepared=prepared,
                anchor_quantile=anchor_quantile,
            )
        except Exception as exc:
            errors[algo] = str(exc)
            continue
        if model is not None:
            models[algo] = model
    if not models and "_geral" not in errors:
        errors["_geral"] = "Falha ao treinar os modelos."
    return models, errors
def make_target(
    train_df: pd.DataFrame,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    anchor_quantile: Optional[float] = None,
) -> pd.Series:
    """
    Retorna a série alvo (y) para treino do modelo:
    - historico: usa QtdAux da base (padrão atual da empresa)
    - ideal: usa carga teórica / horas_disp, com fallback para QtdAux se não houver carga
    """
    y_hist = pd.to_numeric(train_df["QtdAux"], errors="coerce")
    if mode == "historico":
        return y_hist
    # mode == "ideal"
    # Alvo ideal sem depender do QtdAux da própria loja.
    # Estima headcount ideal ancorando em Receita/Aux das lojas de melhor performance.
    anchor_q = anchor_quantile
    if anchor_q is None:
        try:
            anchor_q = safe_float(getattr(st.session_state, "anchor_rpa_quantile", 0.60), 0.60)
        except Exception:
            anchor_q = 0.60
    anchor_q = max(0.1, min(0.95, float(anchor_q)))
    horas_disp_safe = max(float(horas_disp), 1e-6)
    receita_raw = train_df.get("ReceitaTotalMes")
    receita_mes = pd.to_numeric(
        receita_raw if isinstance(receita_raw, pd.Series) else pd.Series(receita_raw, index=train_df.index),
        errors="coerce",
    )
    faturamento_raw = train_df.get("Faturamento/Hora")
    faturamento_hora = pd.to_numeric(
        faturamento_raw if isinstance(faturamento_raw, pd.Series) else pd.Series(faturamento_raw, index=train_df.index),
        errors="coerce",
    )
    horas_raw = train_df.get("HorasOperacionais")
    horas_op = pd.to_numeric(
        horas_raw if isinstance(horas_raw, pd.Series) else pd.Series(horas_raw, index=train_df.index),
        errors="coerce",
    )
    horas_op = horas_op.where(horas_op > 0, np.nan)
    receita_est_hora = faturamento_hora * horas_op * WEEKS_PER_MONTH
    base_ativa = pd.to_numeric(train_df.get("BaseAtiva"), errors="coerce")
    receita_est_base = base_ativa * 0.0
    receita_base = receita_mes if isinstance(receita_mes, pd.Series) else pd.Series(receita_mes, index=train_df.index)
    receita_base = receita_base.where(receita_base.notna() & (receita_base > 0), receita_est_hora)
    receita_base = receita_base.where(receita_base.notna() & (receita_base > 0), receita_est_base)

    # Referência de Receita por Auxiliar das lojas de melhor desempenho
    # Âncora (percentil) de Receita/Aux para evitar subdimensionar ou superdimensionar
    anchor_rpa = _get_high_perf_anchor(train_df, q=anchor_q)
    if anchor_rpa is None or anchor_rpa <= 0:
        # fallback: mediana de ReceitaPorAux
        receita_por_aux = _compute_receita_por_aux(train_df, y_hist)
        receita_por_aux = receita_por_aux.replace([np.inf, -np.inf], np.nan)
        anchor_rpa = float(receita_por_aux.median(skipna=True)) if receita_por_aux.notna().any() else np.nan
    if anchor_rpa is None or not np.isfinite(anchor_rpa) or anchor_rpa <= 0:
        anchor_rpa = 1.0

    # Headcount ideal = receita esperada / receita por aux de referência
    y_ideal = receita_base / max(anchor_rpa, 1e-6)
    y_ideal = y_ideal * (1.0 + margem)
    # Se faltar receita, preenche com mediana do próprio alvo ideal (evita usar QtdAux individual)
    mediana_ideal = float(y_ideal.median(skipna=True)) if y_ideal.notna().any() else float(y_hist.median(skipna=True))
    y_ideal = y_ideal.fillna(mediana_ideal)
    # ancora no desempenho das lojas com maior receita por auxiliar (mantém teto de eficiência)
    y_ideal = _apply_high_perf_cap(y_ideal, train_df, margem, anchor_quantile=anchor_q)
    return y_ideal.clip(lower=0.0)
def _compute_receita_por_aux(train_df: pd.DataFrame, qtd_aux: pd.Series) -> pd.Series:
    """Retorna série com receita (mês ou hora) dividida por auxiliar."""
    if train_df is None or qtd_aux is None or train_df.empty:
        return pd.Series(dtype="float64")
    qaux = pd.to_numeric(qtd_aux, errors="coerce").replace(0, np.nan)
    base = pd.Series(np.nan, index=train_df.index, dtype="float64")
    if "ReceitaPorAux" in train_df.columns:
        base = pd.to_numeric(train_df["ReceitaPorAux"], errors="coerce")
    if base.notna().sum() < 3 and "ReceitaTotalMes" in train_df.columns:
        receita_mes = pd.to_numeric(train_df["ReceitaTotalMes"], errors="coerce")
        base = receita_mes / qaux
    if (base.notna().sum() < 3) and "Faturamento/Hora" in train_df.columns:
        faturamento_hora = pd.to_numeric(train_df["Faturamento/Hora"], errors="coerce")
        horas_raw = train_df.get("HorasOperacionais")
        if horas_raw is None:
            fator_tempo = 1.0
        else:
            horas = pd.to_numeric(horas_raw, errors="coerce")
            if not isinstance(horas, pd.Series):
                horas = pd.Series(horas, index=train_df.index, dtype="float64")
            horas = horas.where(horas > 0, np.nan)
            fator_tempo = horas.fillna(1.0)
        base = (faturamento_hora * fator_tempo) / qaux
    return base.replace([np.inf, -np.inf], np.nan)
def predict_qtd_auxiliares(
    model: Pipeline,
    feature_row: Dict[str, object],
    *,
    with_queue_adjustment: bool = True,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Executa o pipeline treinado em uma nova linha de features sem aplicar pos-processamento da fila.
    O parametro with_queue_adjustment foi mantido apenas para compatibilidade e nao altera mais o
    resultado; a teoria das filas agora e apresentada como um terceiro modelo independente.
    """
    if model is None:
        raise ValueError("Modelo nao treinado.")
    model_name = model.__class__.__name__.lower()
    is_catboost = isinstance(model, CatBoostQuantileModel) or ("catboost" in model.__class__.__module__.lower()) or (
        "catboost" in model_name
    )
    if not is_catboost:
        raise ValueError("Apenas CatBoost é suportado.")
    pred_result = predict_catboost(model, feature_row)
    pred = float(pred_result.get("pred", np.nan))
    if not math.isfinite(pred):
        raise ValueError("Predicao invalida (NaN/inf).")
    base_pred = max(0.0, pred)
    queue_diag: Optional[Dict[str, float]] = None
    if return_details:
        queue_inputs = estimate_queue_inputs(feature_row)
        fila_diag = diagnosticar_fila(
            queue_inputs["lambda_hora"],
            queue_inputs["tma_min"],
            max(base_pred, 1e-6),
        )
        queue_diag = {
            "tag": "central",
            "capacity": float(base_pred),
            "lambda_hora": float(queue_inputs["lambda_hora"]),
            "tma_min": float(queue_inputs["tma_min"]),
            "mu_hora": float(fila_diag["mu_hora"]),
            "rho": float(fila_diag["rho"]),
        }
    if return_details:
        return base_pred, (queue_diag or {})
    return base_pred
def vif_via_aux_regressions(df: pd.DataFrame, cols: List[str]) -> List[Tuple[str, float]]:
    """Estima o VIF ajustando regressões auxiliares entre as features."""
    from sklearn.linear_model import LinearRegression
    X = df[[c for c in cols if c in df.columns]].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(how="all", axis=1)
    if X.shape[1] == 0: return []
    X = X.fillna(X.median(numeric_only=True))
    Xv = X.to_numpy(dtype=float)
    names = list(X.columns)
    out = []
    for j, name in enumerate(names):
        yj = Xv[:, j]
        X_ = np.delete(Xv, j, axis=1)
        if X_.shape[1] == 0:
            out.append((name, 1.0))
            continue
        reg = LinearRegression()
        reg.fit(X_, yj)
        r2 = reg.score(X_, yj)
        r2 = float(np.clip(r2, 0.0, 0.999999))
        vif = 1.0 / (1.0 - r2)
        out.append((name, float(vif)))
    return out
def collinearity_report(df: pd.DataFrame, cols: List[str], corr_thr: float = 0.85) -> Dict[str, object]:
    """Gera métricas e alertas de multicolinearidade para as features."""
    X = df[[c for c in cols if c in df.columns]].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(how="all", axis=1).fillna(X.median(numeric_only=True))
    corr = X.corr(numeric_only=True)
    high_corr_pairs = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j <= i: continue
            r = corr.loc[c1, c2]
            if np.isfinite(r) and abs(r) >= corr_thr:
                high_corr_pairs.append((c1, c2, float(r)))
    vif_list = vif_via_aux_regressions(X, list(X.columns))
    return {"corr_matrix": corr, "high_corr_pairs": high_corr_pairs, "vif": vif_list}
def smape(y_true, y_pred, eps=1e-6):
    """Calcula o SMAPE entre valores reais e previstos."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    return np.mean(np.abs(y_pred - y_true) / denom)
def _mean_absolute_percentage_error_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o MAPE ignorando observações cujo valor real é zero para evitar divisão por zero."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not mask.any():
        return float("nan")
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(ape))
def evaluate_trained_model(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    """Calcula m?tricas padr?o e avisos de sanidade para um pipeline treinado."""
    warnings_list: List[str] = []
    preds_list: List[float] = []
    for _, row in X_test.iterrows():
        try:
            val = predict_qtd_auxiliares(
                model,
                row.to_dict(),
                with_queue_adjustment=False,
                return_details=False,
            )
        except Exception:
            val = np.nan
        preds_list.append(val)
    preds_np = np.asarray(preds_list, dtype=float)
    if not np.all(np.isfinite(preds_np)):
        warnings_list.append("Predicoes possuem valores NaN ou infinitos.")
    y_true_np = y_test.to_numpy(dtype=float) if isinstance(y_test, pd.Series) else np.asarray(y_test, dtype=float)
    mask = np.isfinite(preds_np) & np.isfinite(y_true_np)
    if not mask.any():
        return {
            "model_name": model_name,
            "warnings": warnings_list + ["Sem amostras validas para avaliacao."],
            "n_test": 0,
        }
    preds_valid = preds_np[mask]
    y_valid = y_true_np[mask]
    r2_val = float("nan")
    if len(np.unique(y_valid)) > 1:
        r2_val = r2(y_valid, preds_valid)
    mae_val = mae(y_valid, preds_valid)
    rmse_val = rmse(y_valid, preds_valid)
    mape_val = mape_safe(y_valid, preds_valid)
    smape_val = smape(y_valid, preds_valid)
    precision = precision_from_mape(mape_val) if np.isfinite(mape_val) else float("nan")
    if np.isfinite(r2_val) and r2_val < -1.0:
        warnings_list.append("R2 abaixo de -1. Pode haver desalinhamento de dados.")
    if np.isfinite(mape_val) and mape_val > 1.0:
        warnings_list.append("MAPE acima de 100%. Erro percentual muito alto.")
    metrics: Dict[str, object] = {
        "model_name": model_name,
        "R2": r2_val if np.isfinite(r2_val) else np.nan,
        "R2_mean": r2_val if np.isfinite(r2_val) else np.nan,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "MAPE": mape_val if np.isfinite(mape_val) else np.nan,
        "SMAPE": smape_val if np.isfinite(smape_val) else np.nan,
        "Precisao": precision if np.isfinite(precision) else np.nan,
        "Precisao_percent": (precision * 100.0) if np.isfinite(precision) else np.nan,
        "n_test": int(len(y_valid)),
    }
    if warnings_list:
        metrics["warnings"] = warnings_list
    return metrics
def evaluate_model_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,  # legado
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    anchor_quantile: Optional[float] = None,
    algo: Optional[str] = "catboost",
) -> Dict[str, object]:
    """Executa avaliacao hold-out para o CatBoost."""
    _ = n_splits
    train_df = clean_training_dataframe(train_df)
    if train_df is None or train_df.empty:
        return {}
    prepared = _prepare_model_data(train_df, mode, horas_disp, margem, anchor_quantile)
    if prepared is None:
        return {}
    X_full, y_full, used_features, sample_weights = prepared
    split = _split_train_test_data(X_full, y_full, sample_weights, test_size=0.25, random_state=42)
    if split is None:
        return {}
    algo_name = (algo or "catboost").lower()
    metrics_map: Dict[str, object] = {}
    try:
        model = train_auxiliares_model(
            train_df,
            mode=mode,
            horas_disp=horas_disp,
            margem=margem,
            algo=algo_name,
            prepared=(split['X_train'], split['y_train'], used_features, split.get('sample_weight_train')),
            anchor_quantile=anchor_quantile,
        )
    except Exception as exc:
        metrics_map[algo_name] = {'error': str(exc)}
    else:
        if model is None:
            metrics_map[algo_name] = {'error': 'Modelo indisponivel para avaliacao.'}
        else:
            metrics_map[algo_name] = evaluate_trained_model(
                MODEL_ALGO_NAMES.get(algo_name, algo_name),
                model,
                split['X_test'],
                split['y_test'],
            )
    if algo is None:
        return metrics_map
    return metrics_map.get(algo_name, {})
def predict_with_uncertainty(
    train_df: pd.DataFrame,
    feature_row: Dict[str, object],
    n_boot: int = 8,
    q: Tuple[float, float] = (5, 95),
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    anchor_quantile: Optional[float] = None,
    algo: str = "catboost",
) -> Dict[str, float]:
    """Aplica quantile regression do CatBoost para estimar o IC de 98% pré-fila."""
    train_df = clean_training_dataframe(train_df)
    if train_df is None or train_df.empty:
        return {}
    prepared_full = _prepare_model_data(train_df, mode, horas_disp, margem, anchor_quantile)
    if prepared_full is None:
        return {}
    model_cat = train_auxiliares_model(
        train_df,
        mode=mode,
        horas_disp=horas_disp,
        margem=margem,
        algo="catboost",
        prepared=prepared_full,
        anchor_quantile=anchor_quantile,
    )
    if not isinstance(model_cat, CatBoostQuantileModel):
        return {}
    try:
        low_raw, mid_raw, high_raw = model_cat.predict_quantiles_from_row(feature_row)
    except Exception:
        return {}
    interval = intervalo_90_catboost(low_raw, mid_raw, high_raw)
    return {
        **interval,
        "ci_label": "Int. 98% (CatBoost pré-fila)",
        "ci_low_raw": interval["ci_low"],
        "ci_high_raw": interval["ci_high"],
        "ci_low_raw_disp": None,
        "ci_high_raw_disp": None,
        "ci_mid_raw_disp": interval.get("ci_mid_disp"),
        "ci_label_raw": None,
    }
# =============================================================================
# Referências e agrupamentos
# =============================================================================
def get_total_reference_values(fIndicadores: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Retorna agregados (Base, ReceitaTotalMes) priorizando a linha Total (Estado=Total e Praça=Total),
    com fallback para a soma das lojas caso a linha não exista.
    """
    if fIndicadores is None or fIndicadores.empty:
        return {}
    df = fIndicadores.copy()
    for col in ["BaseAtiva", "BaseTotal", "Base", "ReceitaTotalMes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = pd.Series([True] * len(df))
    has_mask = False
    for col in ["Estado", "Praça", "PraÇõa", "Praca"]:
        if col in df.columns:
            mask &= df[col].astype(str).str.strip().str.upper() == "TOTAL"
            has_mask = True
    refs = {}
    if has_mask:
        total_row = df.loc[mask]
        if not total_row.empty:
            row = total_row.iloc[0]
            for col in ["BaseAtiva", "ReceitaTotalMes"]:
                if col in row:
                    val = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(val):
                        key = "BaseAtivaTotal" if col == "BaseAtiva" else col
                        refs[key] = float(val)
    if not refs:
        if "BaseAtiva" in df.columns:
            total_val = pd.to_numeric(df["BaseAtiva"], errors="coerce").sum(min_count=1)
            if pd.notna(total_val) and total_val > 0:
                refs["BaseAtivaTotal"] = float(total_val)
        elif "Base" in df.columns:
            total_val = pd.to_numeric(df["Base"], errors="coerce").sum(min_count=1)
            if pd.notna(total_val) and total_val > 0:
                refs["BaseAtivaTotal"] = float(total_val)
        if "ReceitaTotalMes" in df.columns:
            total_val = pd.to_numeric(df["ReceitaTotalMes"], errors="coerce").sum(min_count=1)
            if pd.notna(total_val) and total_val > 0:
                refs["ReceitaTotalMes"] = float(total_val)
    return refs
def estimate_cluster_indicators(
    fIndicadores: Optional[pd.DataFrame],
    feature_row: Optional[Dict[str, float]],
    cluster_features: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Clusteriza lojas históricas e retorna médias esperadas dos indicadores alvo
    para a loja descrita em feature_row.
    """
    if (
        fIndicadores is None
        or fIndicadores.empty
        or feature_row is None
        or len(feature_row) == 0
    ):
        return {}
    default_features = [
        "BaseTotal",
        "BaseAtiva",
        "ReceitaTotalMes",
        "AtividadeER",
        "Inicios",
        "Reinicios",
        "Recuperados",
        "I4aI6",
    ]
    default_targets = [
        "%Retirada",
        "Faturamento/Hora",
        "Pedidos/Hora",
        "Pedidos/Dia",
        "Itens/Pedido",
    ]
    cluster_features = cluster_features or default_features
    target_cols = target_cols or default_targets
    df = fIndicadores.copy()
    for col in set(cluster_features + target_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # remove linha Total agregada
    mask_total = pd.Series([True] * len(df))
    has_total = False
    for col in ["Estado", "Praça", "Praca"]:
        if col in df.columns:
            mask_total &= df[col].astype(str).str.strip().str.upper() == "TOTAL"
            has_total = True
    if has_total:
        df = df.loc[~mask_total]
    available_features = [c for c in cluster_features if c in df.columns]
    available_targets = [c for c in target_cols if c in df.columns]
    if len(available_features) == 0 or len(available_targets) == 0:
        return {}
    df = df.dropna(subset=available_features)
    if len(df) < 2:
        return {}
    n_clusters = min(4, len(df))
    if n_clusters < 2:
        return {}
    scaler = StandardScaler()
    X = scaler.fit_transform(df[available_features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df["_cluster"] = labels
    cluster_stats = df.groupby("_cluster")[available_targets].mean()
    cluster_input = []
    for col in available_features:
        val = feature_row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = df[col].median()
        cluster_input.append(float(val))
    cluster_label = int(kmeans.predict(scaler.transform([cluster_input]))[0])
    if cluster_label not in cluster_stats.index:
        return {}
    result = cluster_stats.loc[cluster_label].to_dict()
    result["cluster_id"] = cluster_label
    result["cluster_size"] = int((df["_cluster"] == cluster_label).sum())
    result["n_clusters"] = n_clusters
    return result
# =============================================================================
# ClusterizaÇõÇœo por porte (peso para previsÇœo de QtdAux)
# =============================================================================
def _compute_porte_cluster_context(
    train_df: pd.DataFrame,
    mode: str,
    horas_disp: float,
    margem: float,
    anchor_quantile: Optional[float],
) -> Optional[Dict[str, object]]:
    """Cria contexto de clusterizaÇõÇœo por porte para reponderar previsÇœes de QtdAux."""
    if train_df is None or train_df.empty:
        return None
    feature_cols = [
        "BaseAtiva",
        "ReceitaTotalMes",
        "Faturamento/Hora",
        "Area Total",
    ]
    available = [c for c in feature_cols if c in train_df.columns]
    if len(available) < 2:
        return None
    df_feat = train_df[available].apply(pd.to_numeric, errors="coerce")
    target = make_target(train_df, mode=mode, horas_disp=horas_disp, margem=margem, anchor_quantile=anchor_quantile)
    target = pd.to_numeric(target, errors="coerce")
    aligned_idx = target.dropna().index
    df_feat = df_feat.loc[aligned_idx]
    target = target.loc[aligned_idx]
    if df_feat.empty or target.empty:
        return None
    df_feat_filled = df_feat.copy()
    medians = df_feat_filled.median(numeric_only=True)
    for col in df_feat_filled.columns:
        df_feat_filled[col] = df_feat_filled[col].fillna(medians.get(col, 0.0))
    n_clusters = min(4, len(df_feat_filled))
    if n_clusters < 2:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat_filled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster_stats = pd.Series(target.values, index=labels).groupby(level=0).median()
    cluster_sizes = pd.Series(1, index=labels).groupby(level=0).sum()
    porte_map: Dict[int, int] = {}
    # Rankea clusters pelo headcount mediano (maior = porte 1)
    ordered = sorted(cluster_stats.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (cid, _) in enumerate(ordered, start=1):
        porte_map[int(cid)] = int(rank)
    thresholds = {}
    for col in ["BaseAtiva", "ReceitaTotalMes", "Faturamento/Hora", "Area Total", "QtdAux"]:
        if col in df_feat.columns:
            thr = df_feat[col].quantile(0.85)
            if pd.notna(thr) and thr > 0:
                thresholds[col] = float(thr)
    return {
        "model": kmeans,
        "scaler": scaler,
        "feature_cols": available,
        "feature_medians": medians.to_dict(),
        "cluster_stats": cluster_stats.to_dict(),
        "cluster_sizes": cluster_sizes.to_dict(),
        "porte_map": porte_map,
        "thresholds": thresholds,
        "global_median": float(target.median(skipna=True)),
    }
def _is_loja_grande(feature_row: Dict[str, object], thresholds: Dict[str, float]) -> bool:
    """Classifica loja como grande com base em Receita/Base/Faturamento/QtdAux (quantil alto) e pelo menos 2 critérios."""
    if not thresholds:
        return False
    passes = 0
    base_ok = False
    rec_ok = False
    qtd_ok = False
    for col, thr in thresholds.items():
        try:
            val = float(feature_row.get(col, 0.0))
        except Exception:
            val = 0.0
        if math.isfinite(val) and val >= thr:
            passes += 1
            if col == "BaseAtiva":
                base_ok = True
            if col == "ReceitaTotalMes":
                rec_ok = True
            if col == "QtdAux":
                qtd_ok = True
    if (base_ok and rec_ok) or (rec_ok and qtd_ok):
        return True
    return passes >= 2
def _assign_porte_cluster(feature_row: Dict[str, object], ctx: Dict[str, object]) -> Tuple[Optional[int], Optional[float], int]:
    """Atribui cluster de porte e retorna (cluster_id, valor_medio_cluster, tamanho_cluster)."""
    model = ctx.get("model")
    scaler = ctx.get("scaler")
    cols = ctx.get("feature_cols") or []
    medians: Dict[str, float] = ctx.get("feature_medians", {}) or {}
    if model is None or scaler is None or not cols:
        return None, None, 0
    vals: List[float] = []
    for col in cols:
        try:
            val = float(feature_row.get(col, medians.get(col, 0.0)))
            if not math.isfinite(val):
                val = medians.get(col, 0.0)
        except Exception:
            val = medians.get(col, 0.0)
        vals.append(float(val))
    try:
        df_feat = pd.DataFrame([vals], columns=cols)
        label = int(model.predict(scaler.transform(df_feat))[0])
    except Exception:
        return None, None, 0
    cluster_stats: Dict[int, float] = ctx.get("cluster_stats", {}) or {}
    cluster_sizes: Dict[int, int] = ctx.get("cluster_sizes", {}) or {}
    cluster_pred = cluster_stats.get(label)
    cluster_size = int(cluster_sizes.get(label, 0) or 0)
    return label, cluster_pred, cluster_size
def _blend_pred_with_cluster(
    pred_val: float,
    cluster_pred: Optional[float],
    is_large: bool,
    cluster_size: int,
    global_median: float,
) -> Tuple[float, float]:
    """Aplica mistura com média do cluster; retorna (pred_ajust, peso_cluster)."""
    if cluster_pred is None or not math.isfinite(cluster_pred):
        cluster_pred = global_median
    if cluster_pred is None or not math.isfinite(cluster_pred):
        return float(pred_val), 0.0
    peso = 0.35
    if is_large:
        peso = 0.60
    if cluster_size and cluster_size <= 2:
        peso *= 0.6
    peso = max(0.0, min(0.8, float(peso)))
    ajust = (float(pred_val) * (1.0 - peso)) + (float(cluster_pred) * peso)
    return ajust, peso
def _apply_cluster_blend(
    feature_row: Dict[str, object],
    pred_val: float,
    cluster_ctx: Optional[Dict[str, object]],
) -> Tuple[float, Optional[Dict[str, object]]]:
    """Retorna previsÇœo ajustada pelo cluster de porte e metadados do ajuste."""
    if cluster_ctx is None:
        return float(pred_val), None
    cluster_id, cluster_pred, cluster_size = _assign_porte_cluster(feature_row, cluster_ctx)
    is_large = _is_loja_grande(feature_row, cluster_ctx.get("thresholds", {}))
    ajust, peso = _blend_pred_with_cluster(
        pred_val,
        cluster_pred,
        is_large,
        cluster_size,
        cluster_ctx.get("global_median", float(pred_val)),
    )
    info = {
        "cluster_id": cluster_id,
        "cluster_pred": cluster_pred,
        "cluster_size": cluster_size,
        "is_large": is_large,
        "weight": peso,
    }
    return ajust, info
# =============================================================================
# Horas e dimensionamento
# =============================================================================
def infer_horas_loja_e_disp(row: pd.Series) -> Tuple[float, float]:
    """
    Extrai (horas_operacionais_loja, %disp em 0-1) a partir de uma linha da estrutura/pessoas.
    """
    if row is None or len(row) == 0:
        return 8.0, 1.0
    horas_loja = None
    if pd.notna(row.get("HoraAbertura")) and pd.notna(row.get("HoraFechamento")):
        try:
            a = pd.Timestamp.combine(pd.Timestamp.today().date(), row["HoraAbertura"])
            f = pd.Timestamp.combine(pd.Timestamp.today().date(), row["HoraFechamento"])
            h = (f - a).total_seconds()/3600.0
            if 0 < h <= 24:
                horas_loja = h
        except Exception:
            pass
    if horas_loja is None and pd.notna(row.get("HorasOperacionais")):
        horas_loja = float(row["HorasOperacionais"])
    if horas_loja is None:
        horas_loja = 8.0  # fallback conservador
    disp = row.get("%disp", 1.0)
    try:
        disp = float(disp)
        if disp > 1.0:  # veio 80→80%
            disp = disp/100.0
        disp = max(0.0, min(1.0, disp))
    except Exception:
        disp = 1.0
    return float(horas_loja), float(disp)
def infer_dias_operacionais(row: Optional[pd.Series], fallback: float = 6.0) -> float:
    """Obtém o número de dias/semana operados pela loja."""
    if row is None or len(row) == 0:
        return fallback
    dias = row.get("DiasOperacionais", fallback)
    try:
        dias_val = float(dias)
        if dias_val <= 0 or math.isnan(dias_val):
            return fallback
        return float(max(1.0, min(7.0, dias_val)))
    except Exception:
        return fallback
def carga_total_horas_loja(
    tempos_processo: pd.DataFrame,
    frequencias: Optional[pd.DataFrame] = None,
    fator_monotonia: float = 1.0
) -> Tuple[pd.DataFrame, float]:
    """
    Junta tempos (Loja, Processo, tempo_medio_min) com frequencias (Loja, Processo, frequencia)
    e retorna detalhe e carga total (em horas) já com fator de monotonia.
    """
    detalhe, carga = calcular_carga_por_processo(
        tempos_processo=tempos_processo,
        frequencias=frequencias,
        fator_monotonia=fator_monotonia
    )
    return detalhe, float(carga)
def calcular_qtd_aux_ideal(
    carga_total_horas: float,
    horas_por_colaborador: float,
    ocupacao_alvo: float = DEFAULT_OCUPACAO_ALVO,
    margem_operacional: float = 0.10,
    absenteismo: float = DEFAULT_ABSENTEISMO,
    sla_buffer: float = DEFAULT_SLA_BUFFER
) -> dict:
    """
    Converte carga (h) → FTE ideal.
      - horas_por_colaborador: horas efetivas/colab/periodo (já aplicadas %disp)
      - ocupacao_alvo: 0–1 (ex.: 0.80)
      - absenteismo: 0–1 (ex.: 0.08)
      - sla_buffer: 0–1  folga p/ picos/SLA
      - margem_operacional: igual ao que você já usa (margem extra do botão)
    Fórmula: FTE = (carga / horas_por_colaborador) / ocupacao_alvo
             FTE *= (1 + absenteismo + sla_buffer + margem_operacional)
    """
    if horas_por_colaborador <= 0:
        raise ValueError("horas_por_colaborador deve ser > 0")
    fte_base = (carga_total_horas / horas_por_colaborador) / max(0.01, ocupacao_alvo)
    fte_ajust = fte_base * (1.0 + absenteismo + sla_buffer + margem_operacional)
    return {
        "carga_total_horas": float(carga_total_horas),
        "horas_por_colaborador": float(horas_por_colaborador),
        "ocupacao_alvo": float(ocupacao_alvo),
        "absenteismo": float(absenteismo),
        "sla_buffer": float(sla_buffer),
        "margem": float(margem_operacional),
        "qtd_aux_ideal": int(math.ceil(max(0.0, fte_ajust)))
    }
def ideal_simplificado_por_fluxo(
    pedidos_hora: float,
    horas_operacionais_loja: float,
    tmedio_min_atendimento: float,
    fator_monotonia: float,
    horas_por_colaborador: float,
    margem_operacional: float,
    ocupacao_alvo: float,
    absenteismo: float,
    sla_buffer: float
) -> dict:
    """
    Constrói uma carga aproximada sem decompor por processo.
    Todas as horas são semanais (horas_operacionais_loja, horas_por_colaborador).
    pedidos_hora é a média de pedidos por hora ao longo da semana.
    Retorna carga_total_horas em horas semanais.
    """
    # pedidos_hora é média semanal (ex: 12 pedidos/hora)
    # horas_operacionais_loja é total semanal (ex: 84h/semana)
    # Resultado: total de pedidos na semana
    pedidos_total_semana = max(0.0, pedidos_hora) * max(0.0, horas_operacionais_loja)
    # Converte tempo médio de minutos para horas e calcula carga total semanal
    carga = (pedidos_total_semana * max(0.0, tmedio_min_atendimento)) / 60.0
    carga *= max(1.0, float(fator_monotonia))
    resultado = calcular_qtd_aux_ideal(
        carga_total_horas=carga,
        horas_por_colaborador=horas_por_colaborador,
        margem_operacional=margem_operacional,
        ocupacao_alvo=ocupacao_alvo,
        absenteismo=absenteismo,
        sla_buffer=sla_buffer
    )
    resultado.update({
        "pedidos_hora_utilizado": float(max(0.0, pedidos_hora)),
        "horas_operacionais_loja": float(max(0.0, horas_operacionais_loja)),
        "tmedio_min_atendimento": float(max(0.0, tmedio_min_atendimento)),
        "fator_monotonia": float(max(1.0, float(fator_monotonia))),
    })
    return resultado


def _filter_total_rows_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask_total = pd.Series([True] * len(df), index=df.index)
    has_total = False
    for col in ["Estado", "Praça", "Praca"]:
        if col in df.columns:
            mask_total &= df[col].astype(str).str.strip().str.upper() == "TOTAL"
            has_total = True
    if has_total:
        df = df.loc[~mask_total]
    return df


def fit_base_ativa_pedidos_dia(fIndicadores: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Ajusta regressao polinomial (grau 2) BaseAtiva -> Pedidos/Dia usando fIndicadores.
    Retorna coeficientes e qualidade do ajuste.
    """
    if fIndicadores is None or fIndicadores.empty:
        return {}
    df = fIndicadores.copy()
    df = _filter_total_rows_indicadores(df)
    if "BaseAtiva" not in df.columns or "Pedidos/Dia" not in df.columns:
        return {}
    df["BaseAtiva"] = pd.to_numeric(df["BaseAtiva"], errors="coerce")
    df["Pedidos/Dia"] = pd.to_numeric(df["Pedidos/Dia"], errors="coerce")
    df = df.dropna(subset=["BaseAtiva", "Pedidos/Dia"])
    df = df[(df["BaseAtiva"] > 0) & (df["Pedidos/Dia"] > 0)]
    if len(df) < 2:
        return {}
    x = df["BaseAtiva"].astype(float).to_numpy()
    y = df["Pedidos/Dia"].astype(float).to_numpy()
    coef_a, coef_b, coef_c = np.polyfit(x, y, deg=2)
    y_pred = (coef_a * x * x) + (coef_b * x) + coef_c
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - (ss_res / ss_tot))
    return {
        "degree": 2,
        "coef_a": float(coef_a),
        "coef_b": float(coef_b),
        "coef_c": float(coef_c),
        "r2": float(r2),
        "n": int(len(df)),
    }


def estimate_pedidos_dia_from_base_ativa(
    base_ativa: float,
    model_params: Optional[Dict[str, float]],
) -> Optional[float]:
    if not model_params:
        return None
    coef_a = model_params.get("coef_a")
    coef_b = model_params.get("coef_b")
    coef_c = model_params.get("coef_c")
    if coef_a is None or coef_b is None or coef_c is None:
        return None
    base_val = safe_float(base_ativa, 0.0)
    pred = (float(coef_a) * base_val * base_val) + (float(coef_b) * base_val) + float(coef_c)
    if not math.isfinite(pred):
        return None
    return float(max(0.0, pred))


def calcular_media_horas_operacionais(
    estrutura_df: Optional[pd.DataFrame],
    default_horas: float = 10.0,
) -> float:
    if estrutura_df is None or estrutura_df.empty:
        return float(default_horas)
    horas_list: List[float] = []
    for _, row in estrutura_df.iterrows():
        horas = None
        if pd.notna(row.get("HoraAbertura")) and pd.notna(row.get("HoraFechamento")):
            try:
                a = pd.to_datetime(row["HoraAbertura"], errors="coerce")
                f = pd.to_datetime(row["HoraFechamento"], errors="coerce")
                if pd.notna(a) and pd.notna(f):
                    h = (f - a).total_seconds() / 3600.0
                    if 0 < h <= 24:
                        horas = float(h)
            except Exception:
                horas = None
        if horas is None and pd.notna(row.get("HorasOperacionais")):
            horas_val = safe_float(row.get("HorasOperacionais"), 0.0)
            if horas_val > 0:
                horas = float(horas_val)
        if horas is None:
            continue
        dias = safe_float(row.get("DiasOperacionais"), 0.0)
        if horas > 24:
            if dias <= 0:
                dias = 6.0
            horas = horas / max(1.0, min(7.0, dias))
        if 0 < horas <= 24:
            horas_list.append(float(horas))
    if not horas_list:
        return float(default_horas)
    return float(np.mean(horas_list))


def estimate_elasticity_base_to_aux(
    train_df: Optional[pd.DataFrame],
    horas_disp: float,
    margem: float,
    anchor_quantile: Optional[float] = None,
) -> Optional[float]:
    """
    Estima elasticidade log-log de BaseAtiva -> headcount ideal (QtdAux).
    Se beta=0.7, queda de 10% em BaseAtiva implica queda de ~7% em headcount ideal.
    """
    if train_df is None or train_df.empty:
        return None
    try:
        y_ideal = make_target(
            train_df,
            mode="ideal",
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
        )
        base = pd.to_numeric(train_df.get("BaseAtiva"), errors="coerce")
        mask = base.notna() & (base > 0) & y_ideal.notna() & (y_ideal > 0)
        if mask.sum() < 5:
            return None
        logx = np.log(base[mask].to_numpy())
        logy = np.log(y_ideal[mask].to_numpy())
        if len(logx) < 2:
            return None
        slope, _ = np.polyfit(logx, logy, deg=1)
        if not math.isfinite(slope):
            return None
        slope = float(np.clip(slope, 0.3, 1.2))
        return slope
    except Exception:
        return None
# =============================================================================
# Carga e leitura de dados
# =============================================================================
def load_csv_path(path: str, schema: Dict[str, str], aliases: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    # Carrega CSV com tentativa explícita de encodings e delimitadores
    if not os.path.exists(path):
        return create_empty_from_schema(schema)
    detected_sep = None
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
            sample = fh.read(2048)
            try:
                detected_sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                pass
    except Exception:
        detected_sep = None
    sep_kwargs = {"sep": detected_sep or ","}
    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                decimal=",",
                true_values=["VERDADEIRO", "SIM", "True", "1"],
                false_values=["FALSO", "NAO", "NÃO", "False", "0"],
                **sep_kwargs,
            )
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    if df is None:
        try:
            df = pd.read_csv(path, sep=sep_kwargs["sep"])
        except Exception:
            return create_empty_from_schema(schema)
    if df.shape[1] == 1:
        # Provável erro de separador; tenta ; explicitamente com os encodings definidos
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(path, sep=";", encoding=enc, decimal=",")
                break
            except Exception:
                continue
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if aliases:
        df = df.rename(columns={col: aliases.get(col, col) for col in df.columns})
    df = _coerce_types(df, schema)
    return df
AMOSTRAS_COL_ALIASES: Dict[str, str] = {
    "Tempo M�dio": "Tempo Médio",
    "Tempo MǸdio": "Tempo Médio",
    "Tempo Medio": "Tempo Médio",
    "Tempo medio": "Tempo Médio",
    "N�mero de Amostras": "Número de Amostras",
    "Nǧmero de Amostras": "Número de Amostras",
    "Numero de Amostras": "Número de Amostras",
    "\ufeffLoja": "Loja",
    "Loja": "Loja",
}
ALIASES_BY_SCHEMA: Dict[str, Dict[str, str]] = {
    "dAmostras": AMOSTRAS_COL_ALIASES,
}
@st.cache_data(show_spinner=False)    
def _load_csv_cached(path: str, schema_name: str, file_version: float) -> pd.DataFrame:
    # mapa de nomes → factories de schema
    """Carrega CSVs do disco usando cache baseado no timestamp do arquivo."""
    schema_map = {
        "dAmostras": get_schema_dAmostras,
        "dEstrutura": get_schema_dEstrutura,
        "dPessoas": get_schema_dPessoas,
        "fFaturamento2": get_schema_fFaturamento2,
        "fIndicadores": get_schema_fIndicadores,
    }
    schema_fn = schema_map[schema_name]
    aliases = ALIASES_BY_SCHEMA.get(schema_name, {})
    return load_csv_path(path, schema_fn(), aliases)
def _load_with_version(path: str, schema_name: str) -> pd.DataFrame:
    """Empacota a leitura do CSV usando o mtime como chave de cache."""
    file_path = Path(path)
    try:
        mtime = file_path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    return _load_csv_cached(path, schema_name, mtime)
# =============================================================================
# Uploads e sessões do app
# =============================================================================
def apply_operacional_defaults_from_lookup(row: Dict[str, object]) -> None:
    """Atualiza valores padrão de dias/horas operacionais no session_state a partir de um lookup."""
    if not row:
        return
    dias_lookup = safe_float(row.get("DiasOperacionais"), 0.0)
    if dias_lookup and dias_lookup > 0:
        dias_norm = int(max(1, min(7, round(dias_lookup))))
        st.session_state["dias_operacionais_loja_form"] = dias_norm
        st.session_state["dias_operacionais_semana"] = dias_norm
    horas_lookup = safe_float(row.get("HorasOperacionais"), 0.0)
    if horas_lookup and horas_lookup > 0:
        dias_ref = int(st.session_state.get("dias_operacionais_loja_form", 6))
        horas_semanais = horas_lookup * dias_ref if horas_lookup <= 24 else horas_lookup
        st.session_state["horas_loja_config"] = horas_semanais
        st.session_state["horas_operacionais_form"] = horas_semanais
def append_and_dedup(base: pd.DataFrame, new: pd.DataFrame, subset_cols: List[str]) -> pd.DataFrame:
    """Acrescenta linhas e remove duplicidades com base em colunas chave."""
    if base is None or base.empty:
        combined = new.copy()
    else:
        combined = pd.concat([base, new], ignore_index=True)
    if subset_cols:
        combined = combined.drop_duplicates(subset=subset_cols, keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")
    return combined.reset_index(drop=True)
def render_append(nome: str, schema_fn, subset_cols):
    """Renderiza o uploader incremental e aplica validação antes de anexar dados."""
    schema = schema_fn()
    up = st.file_uploader(f"Upload CSV para acrescentar em {nome}", type=["csv"], key=f"up_append_{nome}")
    if up is not None:
        df_up = read_csv_with_schema(up, schema)
        ok, errs = validate_df(df_up, schema)
        if ok:
            st.session_state[nome] = append_and_dedup(st.session_state[nome], df_up, subset_cols)
            if nome == "fIndicadores":
                st.session_state[nome] = _standardize_cols(st.session_state[nome])
            st.success(f"{nome} atualizado. Linhas totais: {len(st.session_state[nome])}")
        else:
            st.error("; ".join(errs))
    st.dataframe(st.session_state[nome].tail(100), use_container_width=True)
# =============================================================================
# Cache de treinamento
# =============================================================================
@st.cache_resource(show_spinner=False)
def _train_cached(
    train_df,
    mode: str,
    horas_disp: float,
    margem: float,
    anchor_quantile: Optional[float] = None,
    cache_version: int = 8,
):
    """Mantém as versões treinadas (um por algoritmo) em cache."""
    models, errors = train_all_auxiliares_models(
        train_df,
        mode=mode,
        horas_disp=horas_disp,
        margem=margem,
        anchor_quantile=anchor_quantile,
    )
    return {"models": models, "errors": errors}
# =============================================================================
# Business logic extracted from app (Streamlit-free helpers)
# =============================================================================
# Movido de app.py: verifica se uma métrica calculada é válida para exibição.
def _metric_has_value(val) -> bool:
    return val is not None and not (isinstance(val, float) and math.isnan(val))
# Movido de app.py: formata os números dos intervalos de confiança exibidos no layout.
def _format_interval_value(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "-"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val)
# Movido de app.py: monta a string resumida com o diagnóstico da fila.
def _format_queue_diag(diag: Optional[Dict[str, float]]) -> str:
    if not diag:
        return ""
    lambda_h = diag.get("lambda_hora")
    tma_min = diag.get("tma_min")
    mu_h = diag.get("mu_hora")
    rho = diag.get("rho")
    capacity = diag.get("capacity")
    parts: List[str] = []
    if _metric_has_value(lambda_h):
        parts.append(f"lambda {lambda_h:.2f}/h")
    if _metric_has_value(tma_min):
        parts.append(f"TMA {tma_min:.1f} min")
    if _metric_has_value(mu_h):
        parts.append(f"mu {mu_h:.2f}/h")
    if _metric_has_value(capacity):
        parts.append(f"c(bruto) {capacity:.2f}")
    if _metric_has_value(rho):
        parts.append(f"rho~{rho:.2f}")
    return " | ".join(parts)
# Movido de app.py: consolida indicadores derivados e resultados de clusterização.
def preparar_indicadores_operacionais(
    base_ativa: float,
    receita_total: float,
    inicios: float,
    reinicios: float,
    recuperados: float,
    i4_a_i6: float,
    total_base_ref: float,
    total_receita_ref: float,
    cluster_targets: List[str],
    indicadores_df: Optional[pd.DataFrame] = None,
    lookup_row: Optional[Dict[str, object]] = None,
    has_lookup: bool = False,
    prefer_manual: bool = False,
) -> Dict[str, object]:
    base_total_den = total_base_ref if total_base_ref and total_base_ref > 0 else None
    receita_total_den = total_receita_ref if total_receita_ref and total_receita_ref > 0 else (receita_total if receita_total > 0 else None)
    pct_base_total = calc_pct(base_ativa, base_total_den)
    pct_faturamento = calc_pct(receita_total, receita_total_den)
    pct_ativos = calc_pct(base_ativa, base_ativa if base_ativa > 0 else None)
    taxa_inicios = calc_pct(inicios, base_ativa if base_ativa > 0 else None)
    taxa_reativacao = calc_pct(recuperados, i4_a_i6)
    taxa_reinicio = calc_pct(reinicios, base_ativa if base_ativa > 0 else None)
    cluster_values = {target: 0.0 for target in cluster_targets}
    cluster_result: Optional[Dict[str, object]] = None
    cluster_used = False
    messages: List[Tuple[str, str]] = []
    if has_lookup and lookup_row and not prefer_manual:
        for target in cluster_targets:
            cluster_values[target] = safe_float(get_lookup(lookup_row, target), 0.0)
    else:
        manual_inputs = [base_ativa, receita_total, inicios, reinicios, recuperados, i4_a_i6]
        manual_has_data = any(val is not None and not pd.isna(val) and float(val) > 0 for val in manual_inputs)
        if manual_has_data:
            cluster_inputs = {
                "BaseAtiva": base_ativa,
                "ReceitaTotalMes": receita_total,
                "Inicios": inicios,
                "Reinicios": reinicios,
                "Recuperados": recuperados,
                "I4aI6": i4_a_i6,
            }
            cluster_result = estimate_cluster_indicators(indicadores_df, cluster_inputs)
            if cluster_result:
                cluster_used = True
                for target in cluster_targets:
                    val = cluster_result.get(target)
                    cluster_values[target] = float(val) if val is not None else 0.0
            else:
                messages.append(
                    (
                        "warning",
                        "Não foi possível estimar os indicadores operacionais por clusterização. Verifique se fIndicadores possui dados suficientes.",
                    )
                )
        else:
            messages.append(
                (
                    "info",
                    "Preencha os indicadores essenciais para estimar os indicadores operacionais por clusterização.",
                )
            )
    return {
        "pct_base_total": pct_base_total,
        "pct_faturamento": pct_faturamento,
        "pct_ativos": pct_ativos,
        "taxa_inicios": taxa_inicios,
        "taxa_reativacao": taxa_reativacao,
        "taxa_reinicio": taxa_reinicio,
        "cluster_values": cluster_values,
        "cluster_result": cluster_result,
        "cluster_used": cluster_used,
        "messages": messages,
    }
# Movido de app.py: monta o dicionário de features usado pelos modelos estatísticos.
def montar_features_input(
    area_total: float,
    qtd_caixas: float,
    horas_operacionais_diarias: float,
    dias_operacionais: float,
    escritorio: int,
    copa: int,
    espaco_evento: int,
    base_ativa: float,
    receita_total_mes: float,
    pct_ativos: float,
    taxa_inicios: float,
    taxa_reativacao: float,
    pedidos_hora: float,
    pedidos_dia: float,
    itens_pedido: float,
    faturamento_hora: float,
    pct_retirada: float,
) -> Dict[str, float]:
    return {
        "Area Total": float(area_total),
        "Qtd Caixas": float(qtd_caixas),
        "HorasOperacionais": float(horas_operacionais_diarias),
        "DiasOperacionais": float(dias_operacionais),
        "Escritorio": int(escritorio),
        "Copa": int(copa),
        "Espaco Evento": int(espaco_evento),
        "BaseAtiva": float(base_ativa),
        "ReceitaTotalMes": float(receita_total_mes),
        "%Ativos": float(pct_ativos),
        "TaxaInicios": float(taxa_inicios),
        "TaxaReativacao": float(taxa_reativacao),
        "Pedidos/Hora": float(pedidos_hora),
        "Pedidos/Dia": float(pedidos_dia),
        "Itens/Pedido": float(itens_pedido),
        "Faturamento/Hora": float(faturamento_hora),
        "%Retirada": float(pct_retirada),
    }
# Movido de app.py: gera dicionários de tempo médio por processo com e sem filtro de loja.
def preparar_dicionarios_tempos_processos(
    amostras_df: Optional[pd.DataFrame],
    loja_nome_alvo: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    tempo_global_dict: Dict[str, float] = {}
    tempo_loja_dict: Dict[str, float] = {}
    if amostras_df is None:
        return tempo_global_dict, tempo_loja_dict
    tempo_global_df = agregar_tempo_medio_por_processo(amostras_df)
    if not tempo_global_df.empty:
        tempo_global_df["Processo_norm"] = tempo_global_df["Processo"].apply(normalize_processo_nome)
        tempo_global_dict = tempo_global_df.groupby("Processo_norm")["tempo_medio_min"].mean().to_dict()
    if loja_nome_alvo:
        amostras_loja_proc, _ = _filter_df_by_loja(amostras_df, loja_nome_alvo)
        tempo_loja_df = agregar_tempo_medio_por_processo(amostras_loja_proc)
        if not tempo_loja_df.empty:
            tempo_loja_df["Processo_norm"] = tempo_loja_df["Processo"].apply(normalize_processo_nome)
            tempo_loja_dict = tempo_loja_df.set_index("Processo_norm")["tempo_medio_min"].to_dict()
    return tempo_global_dict, tempo_loja_dict
# Movido de app.py: estima frequências semanais por processo para o modo simulado.
def calcular_freq_processos_simulacao(
    sim_pedidos_dia: float,
    pedidos_dia_hist: float,
    pedidos_hora_hist: float,
    horas_operacionais_semanais: float,
    dias_operacionais: int,
    base_ativa: float,
    recuperados: float,
    inicios: float,
    reinicios: float,
    itens_pedido_hist: float,
    pct_retirada_hist: float,
    sim_itens_pedido: float,
    sim_pct_retirada: float,
) -> Tuple[float, Dict[str, float]]:
    pedidos_semana_ui = 0.0
    if sim_pedidos_dia > 0:
        pedidos_semana_ui = sim_pedidos_dia * dias_operacionais
    elif pedidos_dia_hist > 0:
        pedidos_semana_ui = pedidos_dia_hist * dias_operacionais
    elif pedidos_hora_hist > 0:
        pedidos_semana_ui = pedidos_hora_hist * max(horas_operacionais_semanais, 1.0)
    auto_freqs = estimate_process_frequencies_from_indicadores(
        base_ativa=base_ativa,
        recuperados=recuperados,
        inicios=inicios,
        reinicios=reinicios,
        pedidos_semana=pedidos_semana_ui,
        itens_por_pedido=sim_itens_pedido if sim_itens_pedido > 0 else itens_pedido_hist,
        pct_retirada=sim_pct_retirada if sim_pct_retirada > 0 else pct_retirada_hist,
        dias_operacionais=dias_operacionais,
    )
    return pedidos_semana_ui, auto_freqs
# Movido de app.py: executa as previsões dos modelos treinados (CatBoost).
def gerar_resultados_modelos(
    model_bundle: Optional[Dict[str, object]],
    train_df: pd.DataFrame,
    features_input: Dict[str, float],
    ref_mode: str,
    horas_disp: float,
    margem: float,
    algo_order: Optional[List[str]] = None,
    anchor_quantile: Optional[float] = None,
    apply_cluster_blend: bool = True,
    compute_metrics: bool = True,
    skip_cap_cols: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    models = (model_bundle or {}).get("models", {}) if model_bundle else {}
    model_errors = dict((model_bundle or {}).get("errors", {})) if model_bundle else {}
    algo_sequence = algo_order or MODEL_ALGO_ORDER
    feature_bounds = _feature_bounds_from_train(train_df)
    capped_features = _cap_feature_row(features_input, feature_bounds, skip_cols=skip_cap_cols)
    cluster_ctx = None
    if apply_cluster_blend and ref_mode == "ideal":
        cluster_ctx = _compute_porte_cluster_context(
            train_df,
            mode=ref_mode,
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
        )
    if not models:
        # mesmo sem modelos carregados, retorne erros conhecidos na ordem esperada
        resultados_stub = []
        for key in algo_sequence:
            if key in model_errors:
                resultados_stub.append(
                    {
                        "key": key,
                        "label": MODEL_ALGO_NAMES.get(key, key),
                        "pred": None,
                        "pred_display": None,
                        "metrics": {},
                        "queue_diag": {},
                        "error": model_errors[key],
                    }
                )
        return resultados_stub, model_errors
    metrics_map: Dict[str, Dict[str, float]] = {}
    if compute_metrics:
        metrics_map = evaluate_model_cv(
            train_df,
            n_splits=5,
            mode=ref_mode,
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
            algo=None,
        )
    resultados: List[Dict[str, object]] = []
    for key in algo_sequence:
        modelo_atual = models.get(key)
        if modelo_atual is None:
            if key in model_errors:
                resultados.append(
                    {
                        "key": key,
                        "label": MODEL_ALGO_NAMES.get(key, key),
                        "pred": None,
                        "pred_display": None,
                        "metrics": {},
                        "queue_diag": {},
                        "error": model_errors[key],
                    }
                )
            continue
        try:
            pred, queue_diag = predict_qtd_auxiliares(
                modelo_atual,
                capped_features,
                return_details=True,
            )
        except Exception as exc:
            model_errors[key] = f"Erro ao prever: {exc}"
            continue
        pred_ajust, cluster_info = (pred, None)
        if cluster_ctx is not None:
            pred_ajust, cluster_info = _apply_cluster_blend(capped_features, pred, cluster_ctx)
        metrics = (metrics_map or {}).get(key, {}) if metrics_map else {}
        resultados.append(
            {
                "key": key,
                "label": MODEL_ALGO_NAMES.get(key, key),
                "pred": pred_ajust,
                "pred_raw": pred,
                "pred_display": max(1.0, round(float(pred_ajust), 2)),
                "metrics": metrics or {},
                "queue_diag": queue_diag or {},
                "cluster_adjust": cluster_info or {},
            }
        )
    return resultados, model_errors
# Movido de app.py: calcula intervalos de confiança para cada algoritmo treinado.
def calcular_intervalos_modelos(
    train_df: pd.DataFrame,
    features_input: Dict[str, float],
    ref_mode: str,
    horas_disp: float,
    margem: float,
    algo_keys: List[str],
    n_boot: int = 8,
    quantis: Tuple[int, int] = (5, 95),
    anchor_quantile: Optional[float] = None,
    apply_cluster_blend: bool = True,
    skip_cap_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    intervalos: Dict[str, Dict[str, float]] = {}
    feature_bounds = _feature_bounds_from_train(train_df)
    capped_features = _cap_feature_row(features_input, feature_bounds, skip_cols=skip_cap_cols)
    cluster_ctx = None
    if apply_cluster_blend and ref_mode == "ideal":
        cluster_ctx = _compute_porte_cluster_context(
            train_df,
            mode=ref_mode,
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
        )
    for key in algo_keys:
        ci = predict_with_uncertainty(
            train_df,
            capped_features,
            n_boot=n_boot,
            q=quantis,
            mode=ref_mode,
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
            algo=key,
        )
        if ci and cluster_ctx is not None:
            pred_base = ci.get("pred_mean", ci.get("ci_mid_disp"))
            pred_base = pred_base if pred_base is not None else 0.0
            pred_adj, info = _apply_cluster_blend(capped_features, pred_base, cluster_ctx)
            peso = (info or {}).get("weight", 0.0)
            cluster_pred = (info or {}).get("cluster_pred", pred_base)
            if info:
                ci["cluster_adjust"] = info
            if cluster_pred is not None and math.isfinite(peso) and peso > 0:
                for field in ["ci_low", "ci_high", "ci_mid_disp", "ci_low_disp", "ci_high_disp", "pred_mean"]:
                    if field in ci and ci[field] is not None and math.isfinite(ci[field]):
                        ci[field] = (float(ci[field]) * (1.0 - peso)) + (float(cluster_pred) * peso)
                ci["pred_mean"] = float(pred_adj)
        if ci:
            intervalos[key] = ci
    return intervalos
# Movido de app.py: consolida horas e dias operacionais da loja para o modo ideal.
def preparar_contexto_operacional(
    loja_nome_alvo: Optional[str],
    estrutura_df: Optional[pd.DataFrame],
    pessoas_df: Optional[pd.DataFrame],
    manual_horas_form: float,
    dias_operacionais_em_uso: int,
    dias_operacionais_ativos: int,
    horas_loja_config: float,
) -> Tuple[float, int]:
    row_horas: Dict[str, object] = {}
    estrutura_row, estrutura_match = _get_loja_row(estrutura_df, loja_nome_alvo)
    pessoas_row, pessoas_match = _get_loja_row(pessoas_df, loja_nome_alvo)
    if estrutura_row:
        row_horas.update(estrutura_row)
    if pessoas_row and "%disp" in pessoas_row:
        row_horas["%disp"] = pessoas_row["%disp"]
    if manual_horas_form > 0:
        if manual_horas_form <= 24:
            horas_diarias_manual = manual_horas_form
        else:
            horas_diarias_manual = manual_horas_form / max(1, dias_operacionais_ativos)
        row_horas["HorasOperacionais"] = horas_diarias_manual
    has_row_data = bool(row_horas)
    if loja_nome_alvo:
        use_loja_dados = bool(estrutura_match or pessoas_match)
    else:
        use_loja_dados = has_row_data
    row_horas["DiasOperacionais"] = dias_operacionais_em_uso
    horas_loja_manual = float(horas_loja_config)
    if horas_loja_manual <= 24:
        horas_loja_manual = horas_loja_manual * dias_operacionais_ativos
    horas_loja = max(float(dias_operacionais_ativos), min(24.0 * dias_operacionais_ativos, horas_loja_manual))
    dias_operacionais_final = dias_operacionais_ativos
    if use_loja_dados and row_horas:
        row_series = pd.Series(row_horas)
        loja_hours, _ = infer_horas_loja_e_disp(row_series)
        dias_row = infer_dias_operacionais(row_series, dias_operacionais_ativos)
        dias_operacionais_final = dias_row
        if pd.notna(loja_hours) and loja_hours > 0:
            horas_loja = float(loja_hours) * dias_operacionais_final
    else:
        horas_raw = float(row_horas.get("HorasOperacionais", 0.0) or horas_loja_manual)
        if horas_raw <= 24:
            horas_raw = horas_raw * dias_operacionais_final
        horas_loja = horas_raw
    if horas_loja <= 0:
        horas_loja = horas_loja_manual
    horas_loja = max(float(dias_operacionais_final), min(24.0 * dias_operacionais_final, horas_loja))
    return float(horas_loja), int(dias_operacionais_final)
# Movido de app.py: avalia a carga por processos antes do cálculo ideal.
def avaliar_carga_operacional_ideal(
    amostras_df: Optional[pd.DataFrame],
    loja_nome_alvo: Optional[str],
    fator_monotonia: float,
    dias_operacionais_ativos: int,
    cluster_values: Dict[str, float],
    horas_loja: float,
    tmedio_min_atend: float,
    horas_por_colab: float,
    margem: float,
    ocupacao_alvo: float,
    absenteismo: float,
    sla_buffer: float,
) -> Dict[str, object]:
    amostras_loja, _ = _filter_df_by_loja(amostras_df, loja_nome_alvo)
    tempos = agregar_tempo_medio_por_processo(amostras_loja)
    # cria DF de frequÇõÇœncias vazio mas com a coluna Loja para evitar KeyError no merge
    freq_df = pd.DataFrame(columns=["Loja", "Processo", "frequencia"])
    detalhe, carga_total_diaria = carga_total_horas_loja(
        tempos_processo=tempos,
        frequencias=freq_df,
        fator_monotonia=fator_monotonia,
    )
    carga_total = carga_total_diaria * dias_operacionais_ativos
    fallback = False
    if carga_total <= 0:
        pedidos_hora_flow = estimate_pedidos_por_hora(
            cluster_values,
            horas_loja,
            dias_operacionais_ativos,
        )
        ideal_simplificado_por_fluxo(
            pedidos_hora=pedidos_hora_flow,
            horas_operacionais_loja=horas_loja,
            tmedio_min_atendimento=tmedio_min_atend,
            fator_monotonia=fator_monotonia,
            horas_por_colaborador=horas_por_colab,
            margem_operacional=margem,
            ocupacao_alvo=ocupacao_alvo,
            absenteismo=absenteismo,
            sla_buffer=sla_buffer,
        )
        fallback = True
    else:
        calcular_qtd_aux_ideal(
            carga_total_horas=carga_total,
            horas_por_colaborador=horas_por_colab,
            margem_operacional=margem,
            ocupacao_alvo=ocupacao_alvo,
            absenteismo=absenteismo,
            sla_buffer=sla_buffer,
        )
    return {
        "detalhe": detalhe,
        "carga_total_diaria": carga_total_diaria,
        "carga_total": carga_total,
        "fallback": fallback,
    }
# Movido de app.py: executa o modo Ideal (Simplificado) completo.
def calcular_resultado_ideal_simplificado(
    cluster_values: Dict[str, float],
    sim_inputs: Dict[str, float],
    horas_loja: float,
    horas_por_colab: float,
    dias_operacionais_ativos: int,
    fator_monotonia: float,
    margem: float,
    sla_buffer: float,
    ocupacao_alvo: float,
    absenteismo: float,
    area_total: float,
    qtd_caixas: float,
    estrutura_flags: Dict[str, int],
    base_ativa: float,
    receita_total: float,
    pct_retirada_hist: float,
    itens_pedido_hist: float,
    faturamento_hora_hist: float,
    processos_freq_dict: Dict[str, float],
    tempo_loja_dict: Dict[str, float],
    tempo_global_dict: Dict[str, float],
) -> Dict[str, object]:
    pedidos_hora_sim = safe_float(cluster_values.get("Pedidos/Hora"), 0.0)
    sim_pedidos_dia = safe_float(sim_inputs.get("pedidos_dia"), 0.0)
    sim_tmedio = safe_float(sim_inputs.get("tmedio_min_atend"), 0.0)
    sim_faturamento_hora = safe_float(sim_inputs.get("faturamento_hora"), 0.0)
    sim_pct_retirada = safe_float(sim_inputs.get("pct_retirada"), pct_retirada_hist)
    sim_itens_pedido = safe_float(sim_inputs.get("itens_pedido"), itens_pedido_hist)
    ticket_medio_ref = 0.0
    cluster_pedidos_hora = safe_float(cluster_values.get("Pedidos/Hora"), 0.0)
    cluster_faturamento_hora = safe_float(cluster_values.get("Faturamento/Hora"), 0.0)
    cluster_pedidos_dia = safe_float(cluster_values.get("Pedidos/Dia"), 0.0)
    if cluster_pedidos_hora > 0 and cluster_faturamento_hora > 0:
        ticket_medio_ref = cluster_faturamento_hora / max(cluster_pedidos_hora, 1e-6)
    elif cluster_pedidos_dia > 0 and cluster_faturamento_hora > 0 and horas_loja > 0:
        faturamento_semana = cluster_faturamento_hora * max(horas_loja, 1.0)
        pedidos_semana = cluster_pedidos_dia * dias_operacionais_ativos
        ticket_medio_ref = faturamento_semana / max(pedidos_semana, 1e-6)
    pedidos_hora_manual = 0.0
    if sim_pedidos_dia > 0 and horas_loja > 0:
        pedidos_semana = sim_pedidos_dia * dias_operacionais_ativos
        pedidos_hora_manual = pedidos_semana / max(horas_loja, 1.0)
    elif sim_faturamento_hora > 0 and ticket_medio_ref > 0:
        pedidos_hora_manual = sim_faturamento_hora / ticket_medio_ref
    pedidos_hora_sim_aj = estimate_pedidos_por_hora(
        cluster_values,
        horas_loja,
        dias_operacionais_ativos,
    )
    fluxo_indicadores = estimate_fluxo_medio_indicadores(
        base_ativa=base_ativa,
        receita_total_mes=receita_total,
        pedidos_dia_hist=cluster_values.get("Pedidos/Dia", 0.0),
        pedidos_hora_hist=cluster_values.get("Pedidos/Hora", 0.0),
        dias_operacionais=dias_operacionais_ativos,
        horas_operacionais_semanais=horas_loja,
    )
    indicador_pedidos_hora = fluxo_indicadores.get("pedidos_hora", 0.0)
    indicador_pedidos_semana = fluxo_indicadores.get("pedidos_semana", 0.0)
    candidatos_fluxo: List[Tuple[float, float]] = []
    if pedidos_hora_manual > 0:
        candidatos_fluxo.append((pedidos_hora_manual, 1.3))
    if pedidos_hora_sim > 0:
        candidatos_fluxo.append((pedidos_hora_sim, 1.0))
    if pedidos_hora_sim_aj > 0:
        candidatos_fluxo.append((pedidos_hora_sim_aj, 0.8))
    if indicador_pedidos_hora > 0:
        candidatos_fluxo.append((indicador_pedidos_hora, 1.1))
    if candidatos_fluxo:
        peso_total = sum(peso for _, peso in candidatos_fluxo)
        pedidos_hora_final = sum(valor * peso for valor, peso in candidatos_fluxo) / max(peso_total, 1e-6)
    else:
        pedidos_hora_final = pedidos_hora_sim_aj
    if pedidos_hora_final <= 0:
        pedidos_hora_final = pedidos_hora_sim_aj
    pedidos_semana_estimado = pedidos_hora_final * max(horas_loja, 1.0)
    if indicador_pedidos_semana > 0:
        if pedidos_semana_estimado <= 0:
            pedidos_semana_estimado = indicador_pedidos_semana
        else:
            pedidos_semana_estimado = 0.5 * pedidos_semana_estimado + 0.5 * indicador_pedidos_semana
    itens_ref = sim_itens_pedido if sim_itens_pedido > 0 else itens_pedido_hist
    itens_ajuste = 1.0 + 0.03 * ((max(itens_ref, 1.0) - 3.0) / 3.0)
    itens_ajuste = max(0.7, min(1.3, itens_ajuste))
    tmedio_base = sim_tmedio if sim_tmedio > 0 else safe_float(sim_inputs.get("tmedio_min_atend"), 0.0)
    tmedio_utilizado = tmedio_base * itens_ajuste
    result_fluxo = ideal_simplificado_por_fluxo(
        pedidos_hora=pedidos_hora_final,
        horas_operacionais_loja=horas_loja,
        tmedio_min_atendimento=tmedio_utilizado,
        fator_monotonia=fator_monotonia,
        horas_por_colaborador=horas_por_colab,
        margem_operacional=margem,
        ocupacao_alvo=ocupacao_alvo,
        absenteismo=absenteismo,
        sla_buffer=sla_buffer,
    )
    area_ref = 250.0
    caixas_ref = 4.0
    estrutura_multiplier = 1.0
    if area_total > 0:
        estrutura_multiplier += min(0.3, (area_total / max(area_ref, 1.0)) * 0.1)
    if qtd_caixas > 0:
        estrutura_multiplier += min(0.1, (qtd_caixas / max(caixas_ref, 1.0)) * 0.05)
    estrutura_multiplier += 0.02 * sum(int(flag) for flag in estrutura_flags.values())
    estrutura_multiplier = max(0.8, min(1.5, estrutura_multiplier))
    pct_retirada_ref = sim_pct_retirada if sim_pct_retirada > 0 else pct_retirada_hist
    faturamento_ref = sim_faturamento_hora if sim_faturamento_hora > 0 else faturamento_hora_hist
    demand_multiplier = 1.0
    demand_multiplier += 0.04 * ((max(pct_retirada_ref, 0.0) - 50.0) / 50.0)
    demand_multiplier += 0.03 * ((max(itens_ref, 1.0) - 3.0) / 3.0)
    if faturamento_ref > 0:
        demand_multiplier += 0.02 * ((faturamento_ref / 800.0) - 1.0)
    demand_multiplier = max(0.75, min(1.6, demand_multiplier))
    carga_fluxo_ajustada = result_fluxo["carga_total_horas"] * estrutura_multiplier * demand_multiplier
    carga_processos_extra = 0.0
    for proc_norm, freq_semana in processos_freq_dict.items():
        if freq_semana is None or freq_semana <= 0:
            continue
        tempo_proc = tempo_loja_dict.get(proc_norm) or tempo_global_dict.get(proc_norm)
        if tempo_proc is None or tempo_proc <= 0:
            continue
        carga_processos_extra += (freq_semana * tempo_proc) / 60.0
    carga_processos_extra *= max(1.0, float(fator_monotonia))
    carga_total_semana = carga_fluxo_ajustada + carga_processos_extra
    horas_por_colab_base = max(1.0, horas_por_colab)
    horas_por_colab_disp = horas_por_colab_base * max(0.1, (1.0 - absenteismo))
    horas_por_colab_disp = horas_por_colab_disp / max(1.0, fator_monotonia)
    horas_por_colab_disp = max(1.0, horas_por_colab_disp)
    carga_por_colab = horas_por_colab_disp * max(0.01, ocupacao_alvo)
    qtd_aux_base = carga_total_semana / max(0.1, carga_por_colab)
    qtd_aux_ajust = qtd_aux_base * (1.0 + margem + sla_buffer)
    result_fluxo_clean = dict(result_fluxo)
    result_fluxo_clean.pop("qtd_aux_ideal", None)
    result_ideal = {
        "carga_total_horas": carga_total_semana,
        "horas_por_colaborador": horas_por_colab_disp,
        "horas_por_colaborador_base": horas_por_colab_base,
        "ocupacao_alvo": ocupacao_alvo,
        "absenteismo": absenteismo,
        "sla_buffer": sla_buffer,
        "margem": margem,
        "qtd_aux_ideal": int(math.ceil(max(0.0, qtd_aux_ajust))),
    }
    flux_base = result_fluxo_clean.get("carga_total_horas", 0.0)
    result_ideal.update(result_fluxo_clean)
    result_ideal["carga_total_horas"] = carga_total_semana
    result_ideal["horas_por_colaborador"] = horas_por_colab_disp
    result_ideal["horas_por_colaborador_base"] = horas_por_colab_base
    result_ideal["carga_fluxo_base"] = flux_base
    result_ideal["carga_fluxo"] = carga_fluxo_ajustada
    result_ideal["estrutura_multiplier"] = estrutura_multiplier
    result_ideal["demand_multiplier"] = demand_multiplier
    result_ideal["carga_processos_extras"] = carga_processos_extra
    result_ideal["pedidos_hora_utilizado"] = pedidos_hora_final
    result_ideal["pedidos_semana_estimado"] = pedidos_semana_estimado
    result_ideal["tmedio_min_atendimento"] = tmedio_utilizado
    result_ideal["fator_monotonia"] = fator_monotonia
    return result_ideal
