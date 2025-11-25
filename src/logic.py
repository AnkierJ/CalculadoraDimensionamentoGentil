import math
import io
import streamlit as st
import base64
from typing import Dict, List, Tuple, Optional
import os
import csv
import unicodedata
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
except ImportError:
    variance_inflation_factor = None

try:
    from catboost import CatBoostRegressor  # type: ignore
except ImportError:
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:
    XGBRegressor = None

# Parâmetros padrão para o modo IDEAL
DEFAULT_OCUPACAO_ALVO = 0.80        # % do tempo produtivo (0–1)
DEFAULT_ABSENTEISMO   = 0.08        # férias + faltas + treinamentos (0–1)
DEFAULT_SLA_BUFFER    = 0.05        # folga extra p/ pico/SLA além da margem

# =============================================================================
# Recursos visuais e normalização
# =============================================================================

def image_to_base64(path):
    """Converte o arquivo indicado em string base64 para embutir imagens no app."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _norm_code(x):
    """Normaliza códigos removendo sufixos numéricos e caracteres não alfanuméricos."""
    s = "" if pd.isna(x) else str(x).strip()
    # remove .0 de floats int-like e lixo não alfanumérico
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isalnum()).upper()
    return s

def safe_float(val, default=0.0):
    """Converte strings com pontuação, %, ou R$ em float seguro usando um default."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        if isinstance(val, str):
            s = val.strip()
            # remove separador de milhar . e converte vírgula para ponto
            s = s.replace(".", "").replace(",", ".")
            # remove símbolos comuns
            s = s.replace("%", "").replace("R$", "").strip()
            if s == "":
                return default
            return float(s)
        return float(val)
    except (ValueError, TypeError):
        return default

SYN = {
    "BaseTotal":        ["BaseTotal", "Base Total", "Base_Total", "Base"],
    "BaseAtiva":        ["BaseAtiva", "Base Ativa", "Base_Ativa"],
    "ReceitaTotalMes":  ["ReceitaTotalMes", "Receita Total", "Receita Total Mes", "Receita_Total_Mes"],
    "ReaisPorAtivo":    ["ReaisPorAtivo", "Reais por Ativo", "Boleto Médio", "Boleto Medio", "Boleto_Medio"],
    "AtividadeER":      ["AtividadeER", "Atividade ER", "Atividade_ER"],
    "Churn":            ["Churn"],
    "A0":               ["A0"],
    "A1aA3":            ["A1aA3", "A1 a A3", "A1_A3"],
    "I4aI6":            ["I4aI6", "I4 a I6", "I4_I6"],
    "Inicios":          ["Inicios", "Inícios", "Inicios (Qtd)"],
    "Reinicios":        ["Reinicios", "Reinícios", "Reinicios (Qtd)"],
    "Recuperados":      ["Recuperados"],
    "%Retirada":        ["%Retirada", "% Retirada", "PctRetirada"],
    "Pedidos/Hora":     ["Pedidos/Hora", "Pedidos por Hora", "Pedidos_Hora"],
    "Pedidos/Dia":      ["Pedidos/Dia", "Pedidos por Dia", "Pedidos_Dia", "Pedidos / Dia"],
    "Itens/Pedido":     ["Itens/Pedido", "Itens por Pedido", "Itens_Pedido"],
    "Faturamento/Hora": ["Faturamento/Hora", "Faturamento por Hora", "Faturamento_Hora"],
    "Area Total":       ["Area Total"],
    "Qtd Caixas":       ["Qtd Caixas", "Caixas"],
    "Espaco Evento":    ["Espaco Evento", "Esp Conv", "Espaco_Conv"],
    "Escritorio":       ["Escritorio"],
    "Copa":             ["Copa"],
    "HorasOperacionais":["HorasOperacionais", "Horas Operacionais"],
}

TRUE_BOOL_VALUES = {"VERDADEIRO","SIM","S","TRUE","T","1","YES","Y"}
FALSE_BOOL_VALUES = {"FALSO","NAO","NÃO","N","FALSE","F","0","NO"}

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

# =============================================================================
# Padronização e percentuais
# =============================================================================

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para nomes canônicos e cria campos derivados como A0aA3."""
    if df is None or df.empty: 
        return df
    rename_map = {}
    for canonical, aliases in SYN.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = canonical
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

# =============================================================================
# Schemas e validação de CSVs
# =============================================================================

def get_schema_dAmostras() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha dAmostras."""
    return {
        "Loja": "string",
        "Processo": "string",
        "Amostra": "int",
        "Minutos": "float",
        "Tempo Médio": "float",  # opcional, pode ser derivado
        "Desvio": "float",       # opcional
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
        "HoraAbertura": "string", 
        "HoraFechamento": "string", 
    }

def get_schema_dPessoas() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha dPessoas."""
    return {
        "Loja": "string",
        "QtdAux": "int",
        "QtdLid": "int",
        "%disp": "float",
    }

def get_schema_fFaturamento() -> Dict[str, str]:
    """Schema com os tipos esperados para a planilha fFaturamento."""
    return {
        "Loja": "string",
        "CodPedido": "int",
        "NomeRevendedora": "string",
        "Papel": "string",
        "DataPedido": "date",
        "HoraPedido": "string",          # formato hh:mm:ss ou hh:mm
        "DataAprovacao": "date",
        "HoraAprovacao": "string",       # manter como string/hh:mm:ss
        "DataAutorizacao": "date",
        "HoraAutorizacao": "string",     # manter como string/hh:mm:ss
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
        "date": "object",  # parse posterior
    }
    df = pd.DataFrame({col: pd.Series(dtype=dtypes[t]) for col, t in schema.items()})
    return df

def _ensure_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """Verifica e retorna quais colunas obrigatórias ainda não existem no DataFrame."""
    return [c for c in required_cols if c not in df.columns]

def _coerce_types(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Converte as colunas do DataFrame para os tipos definidos no schema."""
    df = df.copy()
    TRUE_VALS = ["VERDADEIRO", "SIM", "S", "TRUE", "T", "1"]
    FALSE_VALS = ["FALSO", "NAO", "NÃO", "N", "FALSE", "F", "0"]
    for col, t in schema.items():
        if col not in df.columns:
            continue
        if t == "string":
            def _to_text(x):
                if pd.isna(x):
                    return ""
                # se veio número, formata sem ".0"
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                if isinstance(x, (float, np.floating)):
                    xi = float(x)
                    if xi.is_integer():
                        return str(int(xi))
                    # remove zeros à direita e ponto se ficar no fim
                    return str(x).rstrip("0").rstrip(".")
                return str(x).strip()
            df[col] = df[col].map(_to_text).astype("object")
        elif t == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t == "boolean":
            if df[col].dtype == 'bool' or str(df[col].dtype).startswith('boolean'):
                df[col] = df[col].astype('boolean')
            else:
                s = df[col].astype(str).str.upper().str.strip()
                df[col] = s.isin(TRUE_VALS).astype('boolean')
        elif t == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dt.date
    return df

def validate_df(df, schema, max_null_frac=0.3):
    """Valida tipos e percentual de nulos de um DataFrame segundo o schema."""
    errors = []
    missing = _ensure_columns(df, list(schema.keys()))
    if missing: errors.append(f"Colunas faltantes: {', '.join(missing)}")
    df_cast = _coerce_types(df, schema)
    for col, t in schema.items():
        if col in df_cast.columns and t in ("float","int","date"):
            frac = df_cast[col].isna().mean()
            if frac > max_null_frac:
                errors.append(f"Valores inválidos em '{col}' (~{frac:.0%})")
    return (len(errors)==0), errors

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
    TRUE_VALS  = list(TRUE_BOOL_VALUES)
    FALSE_VALS = list(FALSE_BOOL_VALUES)
    string_cols = [c for c, t in schema.items() if t == "string"]
    dtype_map = {c: "object" for c in string_cols}
    # Robustez para CSVs com ; e vírgula decimal
    try:
        df = pd.read_csv(
            file_obj,
            sep=";",                # separador correto
            encoding="utf-8-sig",   # remove BOM
            decimal=",",            # vírgula decimal
            true_values=TRUE_VALS,  # valores aceitos como True
            false_values=FALSE_VALS,# valores aceitos como False
            skipinitialspace=True,  # ignora espaço após ;
            dtype=dtype_map,        # preserva 0 a esquerda;
        )
    except Exception:
        df = pd.read_csv(file_obj)
    df.columns = [str(c).strip() for c in df.columns]
    df = _coerce_types(df, schema)
    return df

# =============================================================================
# Busca por loja
# =============================================================================

def _normalize_loja_key(value) -> str:
    """Normaliza nomes de loja removendo diferenças de caixa e acentuação."""
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
    """Garante a existência de uma coluna normalizada de loja para joins confiáveis."""
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

def _get_loja_row(df: pd.DataFrame, loja_nome: str) -> Tuple[Dict[str, object], bool]:
    """Localiza a linha da loja e indica se o match foi exato."""
    if df is None or df.empty or "Loja" not in df.columns:
        return {}, False
    if loja_nome:
        norm_target = _normalize_loja_key(loja_nome)
        series_norm = df["Loja"].astype(str).str.strip().str.casefold()
        mask = series_norm == norm_target
        if mask.any():
            return df.loc[mask].iloc[0].to_dict(), True
    return df.iloc[0].to_dict(), False

def _filter_df_by_loja(df: pd.DataFrame, loja_nome: str) -> Tuple[pd.DataFrame, bool]:
    """Filtra um DataFrame pela loja alvo usando a chave normalizada."""
    if df is None or df.empty or not loja_nome or "Loja" not in df.columns:
        return df, False
    norm_target = _normalize_loja_key(loja_nome)
    series_norm = df["Loja"].astype(str).str.strip().str.casefold()
    mask = series_norm == norm_target
    if mask.any():
        return df.loc[mask].copy(), True
    return df, False

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
    Cria uma tabela de frequências por (Loja, Processo) a partir de dAmostras.
    Usa a coluna 'Numero de Amostras' (ou equivalentes) se existir; caso contrário,
    usa a contagem/número de amostras coletadas como proxy.
    """
    if amostras is None or amostras.empty:
        return pd.DataFrame(columns=["Loja", "Processo", "frequencia"])

    df = amostras.copy()
    freq_col = None
    for col in df.columns:
        norm = str(col).strip().lower()
        if "numero" in norm and "amostra" in norm:
            freq_col = col
            break

    group_cols = ["Loja", "Processo"]
    if freq_col:
        freq_df = df.groupby(group_cols, dropna=False)[freq_col].max().reset_index()
        freq_df = freq_df.rename(columns={freq_col: "frequencia"})
    else:
        target_col = "Amostra" if "Amostra" in df.columns else None
        if target_col:
            freq_df = df.groupby(group_cols, dropna=False)[target_col].nunique().reset_index(name="frequencia")
        else:
            freq_df = df.groupby(group_cols, dropna=False).size().reset_index(name="frequencia")

    freq_df["frequencia"] = pd.to_numeric(freq_df["frequencia"], errors="coerce").fillna(0)
    return freq_df

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
            "ReaisPorAtivo",
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
            df[c] = pd.to_numeric(df[c], errors="coerce")
    extra_numeric = [
        "BaseTotal",
        "BaseAtiva",
        "ReceitaTotalMes",
        "Inicios",
        "Reinicios",
        "Recuperados",
        "I4aI6",
        "A0",
        "A1aA3",
        "Churn",
        "AtividadeER",
        "ReceitaPorAux",
    ]
    for extra_col in extra_numeric:
        if extra_col in df.columns:
            df[extra_col] = pd.to_numeric(df[extra_col], errors="coerce")

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

    if "ReaisPorAtivo" not in df.columns:
        df["ReaisPorAtivo"] = pd.NA
    _fill_ratio("ReaisPorAtivo", "ReceitaTotalMes", "BaseAtiva")

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
    for c in [col for col in FEATURE_COLUMNS if c in df.columns]:
        if df[c].isna().all():
            df[c] = 0.0

    if "Loja_norm" in df.columns:
        df = df.drop(columns=["Loja_norm"])
    return df

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
    return cleaned

FEATURE_COLUMNS = [
    # estrutura física
    "Area Total", "Qtd Caixas",
    "Escritorio", "Copa", "Espaco Evento",
    # demanda/fluxo
    "Pedidos/Hora", "Pedidos/Dia", "Itens/Pedido", "Faturamento/Hora", "%Retirada",
    # base comercial
    "BaseAtiva", "ReaisPorAtivo", "%Ativos", "TaxaInicios", "TaxaReativacao",
    # disponibilidade/operacao
    "HorasOperacionais",
]

CONT = ["Area Total","Qtd Caixas","Pedidos/Hora","Pedidos/Dia",
        "Itens/Pedido","Faturamento/Hora","%Retirada","BaseAtiva","ReaisPorAtivo",
        "%Ativos","TaxaInicios","TaxaReativacao","HorasOperacionais"]

MODEL_ALGO_ORDER = ["elasticnet", "catboost", "xgboost", "hgb"]
MODEL_ALGO_NAMES = {
    "elasticnet": "ElasticNetCV",
    "catboost": "CatBoostRegressor",
    "xgboost": "XGBoostRegressor",
    "hgb": "HistGradientBoostingRegressor",
}

def _prepare_model_data(
    train_df: pd.DataFrame,
    mode: str,
    horas_disp: float,
    margem: float,
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

    used_features = drop_high_correlation(train_df, used_features, thr=0.85)
    X = train_df[used_features].copy()
    y = make_target(train_df, mode=mode, horas_disp=horas_disp, margem=margem)
    y = pd.to_numeric(y, errors="coerce")

    mask_valid = y.notna()
    X, y = X.loc[mask_valid], y.loc[mask_valid]
    if X.empty:
        return None

    X, used_features = _reduce_features_by_mi(X, y, used_features)

    for c in X.columns:
        if X[c].dtype.kind in ("b",):
            X[c] = X[c].astype(int)
        X[c] = pd.to_numeric(X[c], errors="coerce")

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

def make_preprocessor_with_pca(used_features: List[str]) -> ColumnTransformer:
    """Monta o ColumnTransformer com imputação, escala e PCA opcional."""
    cont = [c for c in used_features if c in CONT]
    oth  = [c for c in used_features if c not in cont]

    cont_steps = [
        ("imp_med", SimpleImputer(strategy="median")),
        ("imp_zero", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ]
    # só adiciona PCA se houver pelo menos 1 coluna contínua
    if len(cont) >= 1:
        cont_steps.append(("pca", PCA(n_components=min(4, len(cont)))))

    return ColumnTransformer(
        transformers=[
            ("cont", Pipeline(steps=cont_steps), cont),
            ("oth",  Pipeline(steps=[("imp_zero", SimpleImputer(strategy="constant", fill_value=0))]), oth),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def drop_high_correlation(df: pd.DataFrame, cols: List[str], thr: float = 0.85) -> List[str]:
    """Remove features altamente correlacionadas para evitar multicolinearidade."""
    X = df[[c for c in cols if c in df.columns]].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(how="all", axis=1)

    if X.shape[1] <= 1:
        return list(X.columns)

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
                    to_drop.add(c1)
                elif null2 > null1:
                    to_drop.add(c2)
                else:
                    to_drop.add(c1 if var_order[c1] < var_order[c2] else c2)

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

def train_auxiliares_model(
    train_df: pd.DataFrame,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algo: str = "elasticnet",
    prepared: Optional[Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]] = None,
) -> Optional[object]:
    """Treina o modelo solicitado (ElasticNet, CatBoost ou XGBoost)."""
    if prepared is None:
        prepared = _prepare_model_data(train_df, mode, horas_disp, margem)
    if prepared is None:
        return None

    X_full, y_full, used_features, sample_weights = prepared
    X = X_full.copy()
    y = y_full.copy()
    if isinstance(sample_weights, pd.Series):
        sample_weights = sample_weights.reindex(X.index).fillna(1.0)
    else:
        sample_weights = pd.Series(1.0, index=X.index, dtype="float64")
    algo = (algo or "elasticnet").lower()

    if algo == "elasticnet":
        pre = make_preprocessor_with_pca(used_features)
        alpha_grid = np.logspace(-3, 2, 40) if len(X) >= 30 else np.logspace(-2, 1, 20)
        model = Pipeline([
            ("pre", pre),
            ("reg", ElasticNetCV(
                l1_ratio=[0.1,0.3,0.5,0.7,0.9],
                alphas=alpha_grid,
                cv=KFold(n_splits=min(5, max(2, len(X)//3)), shuffle=True, random_state=42),
                max_iter=10000,
                fit_intercept=True,
                random_state=42
            )),
        ])
        fit_params = {"reg__sample_weight": sample_weights.to_numpy(dtype=float)}
        model.fit(X, y, **fit_params)
        model.feature_names_ = list(used_features)
        model.model_feature_names_ = list(used_features)
        return model

    fill_values = X.median(numeric_only=True)
    X_filled = X.fillna(fill_values)
    sample_weight_np = sample_weights.to_numpy(dtype=float)

    if algo == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoostRegressor indisponível. Instale 'catboost'.")
        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            n_estimators=400,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_filled.astype(np.float32), y, sample_weight=sample_weight_np)
        model.model_feature_names_ = list(used_features)
        model.fill_values_ = fill_values
        return model

    if algo == "hgb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=3,
            max_iter=250,
            l2_regularization=0.1,
            min_samples_leaf=3,
            random_state=42,
        )
        model.fit(X_filled.astype(np.float64), y, sample_weight=sample_weight_np)
        model.model_feature_names_ = list(used_features)
        model.fill_values_ = fill_values
        return model

    if algo == "xgboost":
        if XGBRegressor is None:
            raise ImportError("XGBoostRegressor indisponível. Instale 'xgboost'.")
        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.2,
            objective="reg:squarederror",
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            tree_method="hist",
        )
        model.fit(X_filled.astype(np.float32), y, sample_weight=sample_weight_np)
        model.model_feature_names_ = list(used_features)
        model.fill_values_ = fill_values
        return model

    raise ValueError(f"Algoritmo '{algo}' não suportado.")

def train_all_auxiliares_models(
    train_df: pd.DataFrame,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algos: Optional[List[str]] = None,
) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Treina todos os modelos disponíveis e retorna (modelos, erros)."""
    algos = algos or MODEL_ALGO_ORDER
    models: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    prepared = _prepare_model_data(train_df, mode, horas_disp, margem)
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
            )
        except Exception as exc:
            errors[algo] = str(exc)
            continue
        if model is not None:
            models[algo] = model

    if not models and "_geral" not in errors:
        errors["_geral"] = "Falha ao treinar os modelos."
    return models, errors

def make_target(train_df: pd.DataFrame,
                mode: str = "historico",
                horas_disp: float = 6.0,
                margem: float = 0.15) -> pd.Series:
    """
    Retorna a série alvo (y) para treino do modelo:
    - historico: usa QtdAux da base (padrão atual da empresa)
    - ideal: usa carga teórica / horas_disp, com fallback para QtdAux se não houver carga
    """
    y_hist = pd.to_numeric(train_df["QtdAux"], errors="coerce")

    if mode == "historico":
        return y_hist

    # mode == "ideal"
    # Se no futuro você criar uma coluna de carga por loja (ex: CargaTeoricaHoras),
    # ela entra aqui. Por enquanto, deixa preparado:
    y_ideal = pd.Series(np.nan, index=train_df.index, dtype="float64")
    if "CargaTeoricaHoras" in train_df.columns:
        carga = pd.to_numeric(train_df["CargaTeoricaHoras"], errors="coerce")
        y_ideal = (carga / horas_disp) * (1.0 + margem)

    y_final = y_ideal.fillna(y_hist)
    receita_por_aux = _compute_receita_por_aux(train_df, y_hist)
    receita_por_aux = receita_por_aux.replace([np.inf, -np.inf], np.nan)
    valid = receita_por_aux.notna() & (receita_por_aux > 0)
    if valid.sum() >= 3:
        ref = receita_por_aux.loc[valid].median()
        if pd.notna(ref) and ref > 0:
            ajuste = (ref / receita_por_aux).clip(lower=0.55, upper=1.45).fillna(1.0)
            y_final = y_final * ajuste.reindex(y_final.index, fill_value=1.0)
    return y_final.clip(lower=0.0)

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

def predict_qtd_auxiliares(model: Pipeline, feature_row: Dict[str, object]) -> float:
    """Executa o pipeline treinado em uma nova linha de features."""
    if model is None:
        raise ValueError("Modelo não treinado.")
    feature_names = getattr(model, "model_feature_names_", None) or getattr(model, "feature_names_", None)
    if not feature_names:
        raise ValueError("Pipeline não contém feature_names_. Re-treine com o novo train_auxiliares_model.")

    # Monta DF com as MESMAS colunas do treino
    row = {c: feature_row.get(c, None) for c in feature_names}
    df = pd.DataFrame([row], columns=feature_names)

    # Converte booleanos conhecidos
    df = _bool_to_int(df, [c for c in ["Escritorio","Copa","Espaco Evento"] if c in df.columns])

    # Força numérico; NaN pode ficar que o Imputer tratará
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fill_vals = getattr(model, "fill_values_", None)
    if fill_vals is not None:
        df = df.fillna(fill_vals)

    pred = float(model.predict(df)[0])
    return max(0.0, pred)

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

def evaluate_model_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algo: str = "elasticnet",
) -> Dict[str, float]:
    """Executa validação cruzada e consolida métricas de erro."""
    from sklearn.model_selection import KFold
    train_df = clean_training_dataframe(train_df)
    if train_df is None or train_df.empty:
        return {}
    used = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    if not used: return {}
    X = train_df[used].copy()
    y = make_target(train_df, mode=mode, horas_disp=horas_disp, margem=margem)
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    if len(X) < 4:
        return {}
    for c in X.columns:
        if X[c].dtype.kind in ("b",): X[c] = X[c].astype(int)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())

    best_splits = min(n_splits, max(2, len(X)//2))
    splitter = KFold(n_splits=best_splits, shuffle=True, random_state=42)
    r2s, mapes, smapes = [], [], []
    for tr, te in splitter.split(X):
        tr_df = X.iloc[tr].copy()
        tr_df["QtdAux"] = y.iloc[tr].values
        teX, tey = X.iloc[te], y.iloc[te]
        model = train_auxiliares_model(tr_df, mode=mode, horas_disp=horas_disp, margem=margem, algo=algo)
        if model is None:
            continue
        preds = []
        for _, row in teX.iterrows():
            try:
                preds.append(predict_qtd_auxiliares(model, row.to_dict()))
            except Exception:
                preds.append(np.nan)
        preds = pd.Series(preds, index=tey.index, dtype="float64")
        valid_mask = preds.notna()
        if not valid_mask.any():
            continue
        tey_valid = tey.loc[valid_mask]
        preds_valid = preds.loc[valid_mask]
        if np.isclose(np.var(tey_valid), 0.0):
            # sem variação -> R2 não faz sentido, pula fold
            mapes.append(mean_absolute_percentage_error(tey_valid, preds_valid))
            smapes.append(smape(tey_valid, preds_valid))
            continue
        r2s.append(r2_score(tey_valid, preds_valid))
        mapes.append(mean_absolute_percentage_error(tey_valid, preds_valid))
        smapes.append(smape(tey_valid, preds_valid))

    if not r2s:
        return {}
    return {
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "MAPE_mean": float(np.mean(mapes)),
        "SMAPE_mean": float(np.mean(smapes)),
        "Precisao_percent": float((1-np.mean(mapes))*100.0)
    }

def predict_with_uncertainty(
    train_df: pd.DataFrame,
    feature_row: Dict[str, object],
    n_boot: int = 200,
    q: Tuple[float,float]=(5,95),
    mode: str = "historico",
    horas_disp: float = 6.0,
    margem: float = 0.15,
    algo: str = "elasticnet",
) -> Dict[str, float]:
    """Aplica bootstrap para estimar previsões e intervalo de confiança."""
    train_df = clean_training_dataframe(train_df)
    if train_df is None or train_df.empty:
        return {}
    preds = []
    for _ in range(n_boot):
        sample = train_df.sample(frac=1.0, replace=True, random_state=None)
        model_b = train_auxiliares_model(sample, mode=mode, horas_disp=horas_disp, margem=margem, algo=algo)
        if model_b is None:
            continue
        try:
            preds.append(predict_qtd_auxiliares(model_b, feature_row))
        except Exception:
            continue
    if not preds:
        return {}
    lo, hi = np.percentile(preds, list(q))
    return {"pred_mean": float(np.mean(preds)), "ci_low": float(lo), "ci_high": float(hi)}

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
    for col in ["BaseTotal", "ReceitaTotalMes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = pd.Series([True] * len(df))
    has_mask = False
    for col in ["Estado", "Praça", "Praca"]:
        if col in df.columns:
            mask &= df[col].astype(str).str.strip().str.upper() == "TOTAL"
            has_mask = True
    refs = {}
    if has_mask:
        total_row = df.loc[mask]
        if not total_row.empty:
            row = total_row.iloc[0]
            for col in ["BaseTotal", "ReceitaTotalMes"]:
                if col in row:
                    val = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(val):
                        refs[col] = float(val)

    if not refs:
        for col in ["BaseTotal", "ReceitaTotalMes"]:
            if col in df.columns:
                total_val = pd.to_numeric(df[col], errors="coerce").sum(min_count=1)
                if pd.notna(total_val) and total_val > 0:
                    refs[col] = float(total_val)

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
        "ReaisPorAtivo",
        "AtividadeER",
        "Inicios",
        "Reinicios",
        "Recuperados",
        "Churn",
        "A0",
        "A1aA3",
        "A0aA3",
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
# Horas e dimensionamento
# =============================================================================

def horas_operacionais_por_colaborador(row: pd.Series) -> float:
    """
    Row pode conter: HoraAbertura, HoraFechamento, %disp (0-1) ou 0-100.
    Se não tiver horas da loja, usa 'HorasOperacionais' se existir.
    """
    horas_loja, disp = infer_horas_loja_e_disp(row)
    return float(horas_loja * disp)

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
    margem_operacional: float = 0.10,
    ocupacao_alvo: float = DEFAULT_OCUPACAO_ALVO,
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

# =============================================================================
# Carga e leitura de dados
# =============================================================================

def load_csv_path(path: str, schema: Dict[str, str]) -> pd.DataFrame:
    # Carrega CSV com tentativas de codificação/sep e normaliza colunas
    """Carrega um CSV local testando encodings e delimitadores diferentes."""
    if not os.path.exists(path):
        return create_empty_from_schema(schema)
    # Tenta detectar delimitador
    detected_sep = None
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as fh:
            sample = fh.read(2048)
            try:
                detected_sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                pass
    except Exception:
        detected_sep = None
    sep_kwargs = {"sep": detected_sep or ","}
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", decimal=",", true_values=["VERDADEIRO","SIM","True","1"], false_values=["FALSO","NAO","NÃO","False","0"], **sep_kwargs)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1", decimal=",", true_values=["VERDADEIRO","SIM","True","1"], false_values=["FALSO","NAO","NÃO","False","0"], **sep_kwargs)
        except Exception:
            return create_empty_from_schema(schema)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return create_empty_from_schema(schema)
    if df.shape[1] == 1:
        # Provável erro de separador; tenta ; explicitamente
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",")
        except Exception:
            try:
                df = pd.read_csv(path, sep=";", encoding="latin-1", decimal=",")
            except Exception:
                pass
    df.columns = [str(c).strip() for c in df.columns]
    df = _coerce_types(df, schema)
    return df

@st.cache_data(show_spinner=False)    
def _load_csv_cached(path: str, schema_name: str, file_version: float) -> pd.DataFrame:
    # mapa de nomes → factories de schema
    """Carrega CSVs do disco usando cache baseado no timestamp do arquivo."""
    schema_map = {
        "dAmostras": get_schema_dAmostras,
        "dEstrutura": get_schema_dEstrutura,
        "dPessoas": get_schema_dPessoas,
        "fFaturamento": get_schema_fFaturamento,
        "fIndicadores": get_schema_fIndicadores,
    }
    schema_fn = schema_map[schema_name]
    return load_csv_path(path, schema_fn())

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

def ensure_dir(path: str) -> None:
    """Garante que o diretório alvo exista antes de salvar arquivos."""
    os.makedirs(path, exist_ok=True)

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
def _train_cached(train_df, mode: str, horas_disp: float, margem: float):
    """Mantém as versões treinadas (um por algoritmo) em cache."""
    models, errors = train_all_auxiliares_models(
        train_df,
        mode=mode,
        horas_disp=horas_disp,
        margem=margem,
    )
    return {"models": models, "errors": errors}
