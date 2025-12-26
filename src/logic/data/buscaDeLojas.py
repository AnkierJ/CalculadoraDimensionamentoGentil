#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Busca por loja
# =============================================================================

# =============================================================================
# Imports
# =============================================================================
from typing import Dict, List, Optional, Tuple
import unicodedata
import pandas as pd


# =============================================================================
# Helpers internos
# =============================================================================
def _normalize_loja_key(value) -> str:
    """Normaliza nomes de loja removendo diferencas de caixa e acentuacao."""
    if value is None:
        return ""
    text = str(value).strip()
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def _ensure_loja_key(df: Optional[pd.DataFrame], key_col: str = "Loja_norm") -> pd.DataFrame:
    """Garante a existencia de uma coluna normalizada de loja para joins confiaveis."""
    if df is None or df.empty:
        return df
    loja_col = None
    for col in df.columns:
        normalized = str(col)
        ascii_norm = normalized.encode("ascii", "ignore").decode().strip()
        if ascii_norm.casefold() == "loja":
            loja_col = col
            break
    if loja_col is None:
        return df
    out = df.copy()
    if loja_col != "Loja":
        out = out.rename(columns={loja_col: "Loja"})
    out[key_col] = out["Loja"].astype(str).apply(_normalize_loja_key)
    return out


# =============================================================================
# API
# =============================================================================
def carregar_lojas(df_lojas: Optional[pd.DataFrame] = None, path_csv: Optional[str] = None) -> pd.DataFrame:
    """Carrega um DataFrame de lojas a partir de um DF existente ou de um CSV e normaliza a chave."""
    df = df_lojas
    if df is None and path_csv:
        try:
            df = pd.read_csv(path_csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(path_csv)
    return _ensure_loja_key(df)


def listar_nomes_lojas(df_lojas: Optional[pd.DataFrame]) -> List[str]:
    """Retorna nomes unicos de lojas ja limpos/normalizados."""
    if df_lojas is None or df_lojas.empty or "Loja" not in df_lojas.columns:
        return []
    nomes = [
        str(loja).strip()
        for loja in df_lojas["Loja"].dropna().tolist()
        if str(loja).strip()
    ]
    return sorted(set(nomes))


def obter_loja_por_nome(df_lojas: Optional[pd.DataFrame], nome_loja: str) -> Tuple[Dict[str, object], bool]:
    """Obtem a linha da loja pelo nome normalizado, com indicador de match exato."""
    if df_lojas is None or df_lojas.empty or "Loja" not in df_lojas.columns:
        return {}, False
    df_norm = _ensure_loja_key(df_lojas)
    norm_target = _normalize_loja_key(nome_loja)
    mask = df_norm["Loja_norm"] == norm_target
    if mask.any():
        return df_norm.loc[mask].iloc[0].to_dict(), True
    return df_norm.iloc[0].to_dict(), False


def filtrar_lojas(df_lojas: Optional[pd.DataFrame], termo: str) -> pd.DataFrame:
    """Filtra lojas cujo nome contem o termo informado (case-insensitive, acento normalizado)."""
    if df_lojas is None or df_lojas.empty or "Loja" not in df_lojas.columns or not termo:
        return df_lojas if df_lojas is not None else pd.DataFrame()
    df_norm = _ensure_loja_key(df_lojas)
    termo_norm = _normalize_loja_key(termo)
    mask = df_norm["Loja_norm"].str.contains(termo_norm, na=False)
    return df_norm.loc[mask].copy()


def _get_loja_row(df: pd.DataFrame, loja_nome: str) -> Tuple[Dict[str, object], bool]:
    """Localiza a linha da loja e indica se o match foi exato (compatibilidade)."""
    return obter_loja_por_nome(df, loja_nome)


def _filter_df_by_loja(df: pd.DataFrame, loja_nome: str) -> Tuple[pd.DataFrame, bool]:
    """Filtra um DataFrame pela loja alvo usando a chave normalizada (compatibilidade)."""
    if df is None or df.empty or not loja_nome or "Loja" not in df.columns:
        return df, False
    df_norm = _ensure_loja_key(df)
    norm_target = _normalize_loja_key(loja_nome)
    mask = df_norm["Loja_norm"] == norm_target
    if mask.any():
        return df_norm.loc[mask].copy(), True
    return df, False
