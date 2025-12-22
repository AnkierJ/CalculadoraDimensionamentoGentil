from __future__ import annotations

import base64
import io
import numpy as np
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

SYN = {
    "BaseTotal": ["BaseTotal", "Base Total", "Base_Total", "Base"],
    "BaseAtiva": ["BaseAtiva", "Base Ativa", "Base_Ativa"],
    "ReceitaTotalMes": ["ReceitaTotalMes", "Receita Total", "Receita Total Mes", "Receita_Total_Mes"],
    "ReaisPorAtivo": ["ReaisPorAtivo", "Reais por Ativo", "Boleto MǸdio", "Boleto Medio", "Boleto_Medio"],
    "AtividadeER": ["AtividadeER", "Atividade ER", "Atividade_ER"],
    # Churn mantido apenas para compatibilidade de leitura; não mais usado no modelo
    "Churn": ["Churn"],
    "A0": ["A0"],
    "A1aA3": ["A1aA3", "A1 a A3", "A1_A3"],
    "I4aI6": ["I4aI6", "I4 a I6", "I4_I6"],
    "Inicios": ["Inicios", "Inícios", "Inicios (Qtd)"],
    "Reinicios": ["Reinicios", "Reiníios", "Reinicios (Qtd)"],
    "Recuperados": ["Recuperados"],
    "%Retirada": ["%Retirada", "% Retirada", "PctRetirada"],
    "Pedidos/Hora": ["Pedidos/Hora", "Pedidos por Hora", "Pedidos_Hora"],
    "Pedidos/Dia": ["Pedidos/Dia", "Pedidos por Dia", "Pedidos_Dia", "Pedidos / Dia"],
    "Itens/Pedido": ["Itens/Pedido", "Itens por Pedido", "Itens_Pedido"],
    "Faturamento/Hora": ["Faturamento/Hora", "Faturamento por Hora", "Faturamento_Hora"],
    "Area Total": ["Area Total"],
    "Qtd Caixas": ["Qtd Caixas", "Caixas"],
    "Espaco Evento": ["Espaco Evento", "Esp Conv", "Espaco_Conv"],
    "Escritorio": ["Escritorio"],
    "Copa": ["Copa"],
    "HorasOperacionais": ["HorasOperacionais", "Horas Operacionais"],
    "DiasOperacionais": ["DiasOperacionais", "Dias Operacionais", "Dias/semana"],
    "DiasOperacionaisMes": [
        "DiasOperacionaisMes",
        "Dias Operacionais Mes",
        "DiasOperacionais/Mes",
        "DiasOperacionais/Mês",
        "DiasOperacionais/Mì",
        "DiasOperacionais/Mi",
    ],
    "%disp": ["%disp", "% disponibilidade", "% disponibilidade operacional"],
    "%absent": ["%absent", "% absent", "%absenteismo"],
}

TRUE_BOOL_VALUES = {"VERDADEIRO", "SIM", "S", "TRUE", "T", "1", "YES", "Y"}
FALSE_BOOL_VALUES = {"FALSO", "NAO", "NǟO", "N", "FALSE", "F", "0", "NO"}


def image_to_base64(path: Union[str, Path]) -> str:
    """Converte o arquivo indicado em string base64 para embutir imagens no app."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _norm_code(x):
    """Normaliza códigos removendo sufixos numéricos e caracteres não alfanuméricos."""
    s = "" if pd.isna(x) else str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isalnum()).upper()
    return s


def normalize_processo_nome(nome: Optional[str]) -> str:
    """Remove acentos e normaliza o identificador textual de processos."""
    if nome is None:
        return ""
    norm = unicodedata.normalize("NFKD", str(nome))
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    return norm.strip().casefold()


def safe_float(val, default: float = 0.0):
    """Converte strings com pontuação, %, ou R$ em float seguro usando um default."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        if isinstance(val, str):
            s = val.strip()
            s = s.replace(".", "").replace(",", ".")
            s = s.replace("%", "").replace("R$", "").strip()
            if s == "":
                return default
            return float(s)
        return float(val)
    except (ValueError, TypeError):
        return default


def get_lookup(row_dict, canonical_key):
    """Busca um valor usando chaves canônicas e a tabela de sinônimos."""
    for k in SYN.get(canonical_key, [canonical_key]):
        if k in row_dict and row_dict[k] is not None:
            return row_dict[k]
    return None


def get_lookup_value(primary_key, fallback_keys=None):
    """Obtém do session_state valores numéricos com fallback de chaves."""
    lookup_row = st.session_state.get("lookup_row")
    keys = [primary_key]
    if fallback_keys:
        keys.extend(fallback_keys)
    for key in keys:
        val = get_lookup(lookup_row, key)
        if val is not None:
            return safe_float(val, 0.0)
    return 0.0


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para nomes canônicos e cria campos derivados como A0aA3."""
    if df is None or df.empty:
        return df
    rename_map = {}
    for canonical, aliases in SYN.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
    df = df.rename(columns=rename_map)
    if "A0" in df.columns or "A1aA3" in df.columns:
        base_series = pd.Series(0, index=df.index, dtype="float64")
        a0_series = pd.to_numeric(df["A0"], errors="coerce").fillna(0) if "A0" in df.columns else base_series
        a1_series = pd.to_numeric(df["A1aA3"], errors="coerce").fillna(0) if "A1aA3" in df.columns else base_series
        df["A0aA3"] = a0_series + a1_series
    return df


def _standardize_row(row: Dict[str, object]) -> Dict[str, object]:
    """Normaliza um dicionário representando uma linha aplicando os sinônimos do schema."""
    if not row:
        return {}
    normalized = dict(row)
    for canonical, aliases in SYN.items():
        for alias in aliases:
            if alias in row and row[alias] is not None:
                normalized[canonical] = row[alias]
                break

    def _to_float(val):
        if val is None:
            return 0.0
        if isinstance(val, str):
            s = val.strip()
            s = s.replace(".", "").replace(",", ".")
            s = s.replace("%", "").replace("R$", "").strip()
            if not s:
                return 0.0
            try:
                return float(s)
            except ValueError:
                return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    a0_val = _to_float(normalized.get("A0"))
    a1_val = _to_float(normalized.get("A1aA3"))
    if a0_val or a1_val:
        normalized["A0aA3"] = a0_val + a1_val
    return normalized


def calc_pct(numerador: float, denominador: float) -> float:
    """Calcula percentuais protegendo contra divisões inválidas."""
    if denominador is None or pd.isna(denominador) or denominador <= 0:
        return 0.0
    numer = 0.0 if numerador is None or pd.isna(numerador) else float(numerador)
    return (numer / float(denominador)) * 100.0


def get_schema_dAmostras() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha dAmostras."""
    return {
        "Loja": "string",
        "Processo": "string",
        "Amostra": "int",
        "Minutos": "float",
        "Tempo Médio": "float",
        "Desvio": "float",
        "Número de Amostras": "int",
    }


def get_schema_dEstrutura() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha dEstrutura."""
    return {
        "Loja": "string",
        "Area Total": "float",
        "Caixas": "int",
        "Esp Conv": "boolean",
        "Copa": "boolean",
        "Escritorio": "boolean",
        "Shopping": "boolean",
        "HorasOperacionais": "float",
        "DiasOperacionais": "float",
        "DiasOperacionaisMes": "float",
    }


def get_schema_dPessoas() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha dPessoas."""
    return {
        "Loja": "string",
        "QtdAux": "int",
        "QtdLid": "int",
        "%disp": "float",
        "%absent": "float",
    }


def get_schema_fFaturamento2() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha fFaturamento2."""
    return {
        "Loja": "string",
        "CodPedido": "int",
        "NomeRevendedora": "string",
        "Papel": "string",
        "DataPedido": "date",
        "HoraPedido": "string",
        "DataAprovacao": "date",
        "HoraAprovacao": "string",
        "DataAutorizacao": "date",
        "HoraAutorizacao": "string",
        "Faturamento": "float",
        "Retirada": "boolean",
        "CicloMarketing": "string",
        "DiaCiclo": "int",
        "Itens": "int",
    }


def get_schema_fIndicadores() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha fIndicadores."""
    return {
        "BCPS": "string",
        "SAP": "string",
        "Estado": "string",
        "Praça": "string",
        "Loja": "string",
        "BaseTotal": "int",
        "BaseAtiva": "int",
        # Churn mantido apenas para parse de bases antigas
        "Churn": "float",
        "ReceitaTotalMes": "float",
        "ReaisPorAtivo": "float",
        "AtividadeER": "float",
        "A0": "int",
        "A1aA3": "int",
        "I4aI6": "int",
        "Inicios": "int",
        "Reinicios": "int",
        "Recuperados": "int",
        "%daBaseTotal": "float",
        "%doFatTotal": "float",
        "%Ativos": "float",
        "A0aA3": "int",
        "TaxaReativacao": "float",
        "TaxaReinicios": "float",
        "%Retirada": "float",
        "Faturamento/Hora": "float",
        "Pedidos/Hora": "float",
        "Pedidos/Dia": "float",
        "Itens/Pedido": "float",
    }


def create_empty_from_schema(schema: Dict[str, str]) -> pd.DataFrame:
    """Cria um DataFrame vazio com colunas e dtypes do schema informado."""
    dtypes = {
        "string": "object",
        "float": "float64",
        "int": "Int64",
        "boolean": "boolean",
        "date": "object",
    }
    df = pd.DataFrame({col: pd.Series(dtype=dtypes[t]) for col, t in schema.items()})
    return df


def _ensure_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Retorna colunas obrigatórias que ainda não existem no DataFrame."""
    return [c for c in required_cols if c not in df.columns]


def _coerce_types(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Converte as colunas do DataFrame para os tipos definidos no schema."""
    df = df.copy()
    true_vals = ["VERDADEIRO", "SIM", "S", "TRUE", "T", "1"]
    false_vals = ["FALSO", "NAO", "NǟO", "N", "FALSE", "F", "0"]
    for col, t in schema.items():
        if col not in df.columns:
            continue
        if t == "string":
            def _to_text(x):
                if pd.isna(x):
                    return ""
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                if isinstance(x, (float, np.floating)):
                    xi = float(x)
                    if xi.is_integer():
                        return str(int(xi))
                    return str(x).rstrip("0").rstrip(".")
                return str(x).strip()
            df[col] = df[col].map(_to_text).astype("object")
        elif t == "float":
            numeric = pd.to_numeric(df[col], errors="coerce")
            mask_nan = numeric.isna()
            if mask_nan.any():
                fallback = df.loc[mask_nan, col].apply(lambda v: safe_float(v, np.nan))
                numeric.loc[mask_nan] = fallback
            df[col] = numeric.astype(float)
        elif t == "int":
            numeric = pd.to_numeric(df[col], errors="coerce")
            # Permite entrada com decimais (ex: "1.0") arredondando para o inteiro mais próximo antes de converter
            df[col] = numeric.round().astype("Int64")
        elif t == "boolean":
            if df[col].dtype == "bool" or str(df[col].dtype).startswith("boolean"):
                df[col] = df[col].astype("boolean")
            else:
                s = df[col].astype(str).str.upper().str.strip()
                df[col] = s.isin(true_vals).astype("boolean")
        elif t == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
    return df


def validate_df(df, schema, max_null_frac: float = 0.3):
    """Valida tipos e percentual de nulos de um DataFrame segundo o schema."""
    errors = []
    missing = _ensure_columns(df, list(schema.keys()))
    if missing:
        errors.append(f"Colunas faltantes: {', '.join(missing)}")
    df_cast = _coerce_types(df, schema)
    for col, t in schema.items():
        if col in df_cast.columns and t in ("float", "int", "date"):
            frac = df_cast[col].isna().mean()
            if frac > max_null_frac:
                errors.append(f"Valores inválidos em '{col}' (~{frac:.0%})")
    return len(errors) == 0, errors


def template_df(schema: Dict[str, str]) -> pd.DataFrame:
    """Gera um template vazio para download seguindo um schema."""
    return create_empty_from_schema(schema)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa o DataFrame em bytes de CSV prontos para download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def read_csv_with_schema(file_obj, schema: Dict[str, str]) -> pd.DataFrame:
    """Lê um CSV enviado e tenta adequá-lo ao schema conhecido."""
    true_vals = list(TRUE_BOOL_VALUES)
    false_vals = list(FALSE_BOOL_VALUES)
    string_cols = [c for c, t in schema.items() if t == "string"]
    dtype_map = {c: "object" for c in string_cols}
    try:
        df = pd.read_csv(
            file_obj,
            sep=";",
            encoding="utf-8-sig",
            decimal=",",
            true_values=true_vals,
            false_values=false_vals,
            skipinitialspace=True,
            dtype=dtype_map,
        )
    except Exception:
        df = pd.read_csv(file_obj)
    df.columns = [str(c).strip() for c in df.columns]
    df = _coerce_types(df, schema)
    return df
