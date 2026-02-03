# =============================================================================
# Imports
# =============================================================================
import math
import zlib
import html
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from sklearn.cluster import KMeans

from src.logic.core.logic import (
    DEFAULT_OCUPACAO_ALVO,
    DEFAULT_ABSENTEISMO,
    MODEL_ALGO_NAMES,
    WEEKS_PER_MONTH,
    _format_interval_value,
    _format_queue_diag,
    _metric_has_value,
    _train_cached,
    _compute_absenteismo_prefill,
    _estimate_prateleiras_from_area,
    _get_prateleiras_lookup,
    apply_operacional_defaults_from_lookup,
    avaliar_carga_operacional_ideal,
    calcular_intervalos_modelos,
    calcular_resultado_ideal_simplificado,
    calcular_media_horas_operacionais,
    estimate_pedidos_dia_from_base_ativa,
    estimate_pedidos_dia_from_receita,
    fit_base_ativa_pedidos_dia,
    fit_receita_pedidos_dia,
    estimate_elasticity_base_to_aux,
    make_target,
    clean_training_dataframe,
    gerar_resultados_modelos,
    get_total_reference_values,
    montar_features_input,
    preparar_contexto_operacional,
    preparar_dicionarios_tempos_processos,
    preparar_indicadores_operacionais,
    prepare_training_dataframe,
    prepare_estoquistas_training_dataframe,
    predict_estoquistas_extratrees,
)
from src.logic.data.buscaDeLojas import _get_loja_row
from src.logic.models.model_catboost import CATBOOST_PARAM_VERSION, get_catboost_feature_importance
from src.logic.utils.helpers import (
    _norm_code,
    _standardize_row,
    get_criterio_mapeado_key,
    get_criterio_mapeado_label,
    get_criterio_mapeado_options,
    MAPEADO_HELPER_TEXT,
    get_lookup,
    get_lookup_value,
    normalize_processo_nome,
    safe_float,
)


# =============================================================================
# Render principal
# =============================================================================
SIMILARITY_FEATURES = [
    "ReceitaTotalMes",
    "BaseAtiva",
    "Area Total",
    "Faturamento/Hora",
    "Pedidos/Dia",
]
ASG_SEG_PEDIDOS_DIA_THRESHOLD = 162.94
ASG_SEG_AREA_TOTAL_THRESHOLD = 250.0


def _suggest_staff_from_pedidos_dia(
    pedidos_dia_val: Optional[float],
    area_total_val: Optional[float],
) -> int:
    pedidos_dia = safe_float(pedidos_dia_val, float("nan"))
    area_total = safe_float(area_total_val, float("nan"))
    if (
        (math.isfinite(pedidos_dia) and pedidos_dia > ASG_SEG_PEDIDOS_DIA_THRESHOLD)
        or (math.isfinite(area_total) and area_total >= ASG_SEG_AREA_TOTAL_THRESHOLD)
    ):
        return 1
    return 0


def _delta_urgency_label(delta_val: Optional[float]) -> str:
    if delta_val is None or (isinstance(delta_val, float) and pd.isna(delta_val)):
        return ""
    try:
        diff = int(round(float(delta_val)))
    except Exception:
        return ""
    abs_diff = abs(diff)
    if abs_diff <= 1:
        label = "Ótimo"
    elif abs_diff <= 4:
        label = "Bom"
    elif abs_diff <= 9:
        label = "Atenção"
    else:
        label = "Alto"
    return label


def _delta_urgency_color(label_text: str) -> str:
    if not isinstance(label_text, str) or not label_text:
        return "#d1d5db"
    label_norm = label_text.strip().lower()
    if "ótimo" in label_norm:
        color = "#2c9a6c"
    elif "bom" in label_norm:
        color = "#4da3f5"
    elif "atenção" in label_norm:
        color = "#f0b429"
    else:
        color = "#d8516d"
    return color


def _render_urgency_badge(delta_val: Optional[float]) -> str:
    label = _delta_urgency_label(delta_val)
    if not label:
        return ""
    color = _delta_urgency_color(label)
    return (
        "<div style='margin-top:4px;font-size:0.85rem;color:#6c6c6c;'>"
        f"<span style='background:{color}22;color:{color};padding:2px 8px;"
        "border-radius:999px;font-weight:700;'>" + label + "</span></div>"
    )


def _format_estoquistas_helper_text(info: Optional[Dict[str, object]]) -> str:
    if not info:
        return "Modelo de Estoquistas indisponivel."
    if info.get("error"):
        return str(info.get("error"))
    parts: List[str] = [f"Modelo: {info.get('model_name', 'ExtraTreesRegressor')}"]
    metrics = info.get("metrics") or {}
    metric_bits = []
    r2_val = metrics.get("r2")
    mae_val = metrics.get("mae")
    mape_val = metrics.get("mape_pct")
    if isinstance(r2_val, (int, float)) and math.isfinite(r2_val):
        metric_bits.append(f"R2={r2_val:.3f}")
    if isinstance(mae_val, (int, float)) and math.isfinite(mae_val):
        metric_bits.append(f"MAE={mae_val:.3f}")
    if isinstance(mape_val, (int, float)) and math.isfinite(mape_val):
        metric_bits.append(f"MAPE={mape_val:.1f}%")
    if metric_bits:
        parts.append("Metricas: " + ", ".join(metric_bits))
    rows = info.get("rows")
    if isinstance(rows, int) and rows > 0:
        parts.append(f"Amostras: {rows}")
    return " | ".join(parts)


def _build_info_icon(title_text: str) -> str:
    if not title_text:
        return ""
    safe_title = html.escape(title_text, quote=True)
    return (
        f" <span title='{safe_title}' "
        "style='margin-left:4px;display:inline-flex;vertical-align:middle;color:#6c6c6c;'>"
        "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
        "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round' class='icon'>"
        "<circle cx='12' cy='12' r='10'></circle>"
        "<path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'></path>"
        "<line x1='12' y1='17' x2='12.01' y2='17'></line>"
        "</svg></span>"
    )


def _calc_aprendizes_by_total(total_colab: float) -> int:
    if not math.isfinite(total_colab) or total_colab < 7:
        return 0
    base = int(math.floor((total_colab - 1) / 20) + 1)
    min_count = int(math.ceil(total_colab * 0.05))
    max_count = int(math.floor(total_colab * 0.15))
    if max_count < min_count:
        max_count = min_count
    return int(max(min(base, max_count), min_count))


def _estimate_total_colabs(
    pessoas_df: Optional[pd.DataFrame],
    indicadores_df: Optional[pd.DataFrame],
    base_ativa_val: Optional[float],
    receita_total_val: Optional[float],
) -> Optional[float]:
    if pessoas_df is None or pessoas_df.empty or "TOTAL" not in pessoas_df.columns:
        return None
    df = pessoas_df[["Loja", "TOTAL"]].copy() if "Loja" in pessoas_df.columns else pessoas_df.copy()
    df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
    df = df[df["TOTAL"].notna() & (df["TOTAL"] > 0)]
    if df.empty:
        return None
    if indicadores_df is not None and not indicadores_df.empty and "Loja" in df.columns:
        ind_cols = [c for c in ["Loja", "BaseAtiva", "ReceitaTotalMes"] if c in indicadores_df.columns]
        if len(ind_cols) > 1:
            ind = indicadores_df[ind_cols].copy()
            df = df.merge(ind, on="Loja", how="left")
    base_ativa = safe_float(base_ativa_val, float("nan"))
    receita_total = safe_float(receita_total_val, float("nan"))
    if math.isfinite(base_ativa) and "BaseAtiva" in df.columns and df["BaseAtiva"].notna().sum() >= 4:
        series = pd.to_numeric(df["BaseAtiva"], errors="coerce")
        bins = pd.qcut(series, q=4, duplicates="drop")
        target_bin = None
        try:
            target_bin = bins[(series - base_ativa).abs().idxmin()]
        except Exception:
            target_bin = None
        if target_bin is not None:
            bucket = df.loc[bins == target_bin, "TOTAL"]
            if bucket.notna().any():
                return float(bucket.mean())
    if math.isfinite(receita_total) and "ReceitaTotalMes" in df.columns and df["ReceitaTotalMes"].notna().sum() >= 4:
        series = pd.to_numeric(df["ReceitaTotalMes"], errors="coerce")
        bins = pd.qcut(series, q=4, duplicates="drop")
        target_bin = None
        try:
            target_bin = bins[(series - receita_total).abs().idxmin()]
        except Exception:
            target_bin = None
        if target_bin is not None:
            bucket = df.loc[bins == target_bin, "TOTAL"]
            if bucket.notna().any():
                return float(bucket.mean())
    return float(df["TOTAL"].mean())


def _resolve_total_colabs_for_aprendizes(
    lookup_row: Optional[Dict[str, object]],
    pessoas_df: Optional[pd.DataFrame],
    indicadores_df: Optional[pd.DataFrame],
    base_ativa_val: Optional[float],
    receita_total_val: Optional[float],
) -> Optional[float]:
    total_lookup = safe_float((lookup_row or {}).get("TOTAL"), float("nan"))
    if math.isfinite(total_lookup) and total_lookup > 0:
        return total_lookup
    return _estimate_total_colabs(pessoas_df, indicadores_df, base_ativa_val, receita_total_val)


def _apply_aprendizes_rule(
    cargo_sugestao: Optional[Dict[str, float]],
    total_colab: Optional[float],
) -> Tuple[Optional[Dict[str, float]], str]:
    if not cargo_sugestao:
        return cargo_sugestao, ""
    total_val = safe_float(total_colab, float("nan"))
    if not math.isfinite(total_val) or total_val <= 0:
        return cargo_sugestao, ""
    aprendizes = _calc_aprendizes_by_total(total_val)
    cargo_sugestao = dict(cargo_sugestao)
    cargo_sugestao["aprend"] = float(aprendizes)
    aux_lider = safe_float(cargo_sugestao.get("aux_lider"), float("nan"))
    asg_val = safe_float(cargo_sugestao.get("asg", 0.0), 0.0)
    total_time = safe_float(cargo_sugestao.get("total_time_comercial"), float("nan"))
    if math.isfinite(aux_lider) and aux_lider >= 0 and math.isfinite(total_time):
        restante = total_time - asg_val - aprendizes
        if restante < 0:
            restante = 0.0
        lideres = restante / (1.0 + aux_lider) if (1.0 + aux_lider) > 0 else 0.0
        auxiliares = lideres * aux_lider
        cargo_sugestao["lideres"] = float(lideres)
        cargo_sugestao["auxiliares"] = float(auxiliares)
        cargo_sugestao["total_restante"] = float(restante)
    helper = (
        "A partir de 7 colab., representatividade mínima de 5% e máxima de 15%. "
        f"Total = {total_val:.0f} colab. → {aprendizes} aprendizes"
    )
    return cargo_sugestao, helper


def _predict_estoquistas_sugestao(
    train_df: Optional[pd.DataFrame],
    feature_row: Dict[str, object],
) -> Tuple[float, str]:
    if train_df is None or train_df.empty:
        return 0.0, "Modelo de Estoquistas indisponivel."
    info = predict_estoquistas_extratrees(train_df, feature_row)
    helper_text = _format_estoquistas_helper_text(info)
    pred_val = info.get("pred") if isinstance(info, dict) else None
    if pred_val is None or not math.isfinite(float(pred_val)):
        return 0.0, helper_text
    return max(0.0, float(pred_val)), helper_text


def _first_col(df: Optional[pd.DataFrame], options: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in options:
        if col in df.columns:
            return col
    return None


def _compute_cargo_suggestion(
    pessoas_df: Optional[pd.DataFrame],
    indicadores_df: Optional[pd.DataFrame],
    total_ideal: float,
    base_ativa_val: Optional[float] = None,
    pedidos_dia_val: Optional[float] = None,
    area_total_val: Optional[float] = None,
    asg_threshold: float = 162.94,
    asg_area_threshold: float = 250.0,
) -> Optional[Dict[str, float]]:
    total_val = safe_float(total_ideal, 0.0)
    if not math.isfinite(total_val) or total_val <= 0:
        return None
    pedidos_dia = safe_float(pedidos_dia_val, float("nan"))
    area_total = safe_float(area_total_val, float("nan"))
    asg_sugerido = 0.0
    if (
        (math.isfinite(pedidos_dia) and pedidos_dia > asg_threshold)
        or (math.isfinite(area_total) and area_total >= asg_area_threshold)
    ):
        asg_sugerido = 1.0
    total_disponivel = total_val - asg_sugerido
    if total_disponivel < 0:
        total_disponivel = 0.0
    lid_col = _first_col(pessoas_df, ["QtdLid", "QtdLideres", "QtdLider"])
    aux_col = _first_col(pessoas_df, ["QtdAux", "QtdAuxiliar", "QtdAuxiliares"])
    aprendiz_col = _first_col(pessoas_df, ["Aprendiz", "Aprendizes"])
    loja_col = _first_col(pessoas_df, ["Loja"])
    if pessoas_df is None or pessoas_df.empty or not lid_col or not aux_col:
        return None
    cols = [lid_col, aux_col]
    if aprendiz_col:
        cols.append(aprendiz_col)
    if loja_col:
        cols.append(loja_col)
    base = pessoas_df[cols].copy()
    for col in cols:
        if col != loja_col:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    lid_vals = base[lid_col]
    aux_vals = base[aux_col]
    if aprendiz_col:
        aprendiz_vals = base[aprendiz_col]
    else:
        aprendiz_vals = pd.Series(0.0, index=base.index, dtype="float64")
    aux_aprend = (aux_vals + aprendiz_vals).replace([float("inf"), float("-inf")], np.nan)
    valid_mask = (lid_vals > 0) & (aux_aprend > 0)

    ratio_base = pd.DataFrame(
        {
            "ratio": (aux_aprend / lid_vals).replace([float("inf"), float("-inf")], np.nan),
            "aux_share": (aux_vals / aux_aprend).replace([float("inf"), float("-inf")], np.nan),
            "aux_lider": (aux_vals / lid_vals).replace([float("inf"), float("-inf")], np.nan),
            "aprend_lider": (aprendiz_vals / lid_vals).replace([float("inf"), float("-inf")], np.nan),
        }
    )
    ratio_base = ratio_base[valid_mask]

    base_ativa = safe_float(base_ativa_val, float("nan"))
    faixa_label = "Global"
    if (
        indicadores_df is not None
        and not indicadores_df.empty
        and loja_col
        and "Loja" in indicadores_df.columns
        and math.isfinite(base_ativa)
        and base_ativa > 0
    ):
        ind = indicadores_df.copy()
        ind_cols = [c for c in ["Loja", "BaseAtiva"] if c in ind.columns]
        if "BaseAtiva" in ind_cols:
            ind = ind[ind_cols]
            ind["Loja"] = ind["Loja"].astype(str).str.strip()
            base["Loja"] = base["Loja"].astype(str).str.strip()
            merged = base[[loja_col]].copy()
            merged["Loja"] = merged[loja_col].astype(str).str.strip()
            merged = merged.merge(ind, on="Loja", how="left")
            merged["BaseAtiva"] = pd.to_numeric(merged["BaseAtiva"], errors="coerce")
            ratio_base = ratio_base.join(merged["BaseAtiva"])
            if ratio_base["BaseAtiva"].notna().sum() >= 4:
                bins = pd.qcut(ratio_base["BaseAtiva"], q=4, duplicates="drop")
                target_bin = None
                try:
                    target_bin = bins[(ratio_base["BaseAtiva"] - base_ativa).abs().idxmin()]
                except Exception:
                    target_bin = None
                if target_bin is not None:
                    ratio_base = ratio_base[bins == target_bin]
                    try:
                        left = float(target_bin.left)
                        right = float(target_bin.right)
                        faixa_label = f"{left:.2f}-{right:.2f}"
                    except Exception:
                        faixa_label = "Faixa BaseAtiva"

    aux_lider = ratio_base["aux_lider"].dropna()
    aprend_lider = ratio_base["aprend_lider"].dropna()
    if aux_lider.empty or aprend_lider.empty:
        return None
    aux_lider_med = float(aux_lider.median())
    aprend_lider_med = float(aprend_lider.median())
    if not math.isfinite(aux_lider_med) or not math.isfinite(aprend_lider_med):
        return None
    denom = 1.0 + aux_lider_med + aprend_lider_med
    if denom <= 0:
        return None
    aprend_share = aprend_lider_med / denom
    if aprend_share < 0:
        aprend_share = 0.0
    aprend = total_disponivel * aprend_share
    restante_sem_aprend = total_disponivel - aprend
    if restante_sem_aprend < 0:
        restante_sem_aprend = 0.0
    lideres = restante_sem_aprend / (1.0 + aux_lider_med)
    auxiliares = lideres * aux_lider_med
    return {
        "lideres": lideres,
        "auxiliares": auxiliares,
        "aprend": aprend,
        "faixa_label": faixa_label,
        "aux_lider": aux_lider_med,
        "aprend_lider": aprend_lider_med,
        "asg": asg_sugerido,
        "total_time_comercial": total_val,
        "total_restante": restante_sem_aprend,
    }


def _norm_col_key(text: object) -> str:
    if text is None:
        return ""
    norm = unicodedata.normalize("NFKD", str(text))
    norm = "".join(ch for ch in norm if ch.isalnum())
    return norm.casefold()


@st.cache_data(show_spinner=False)
def _load_custo_por_cargo_df() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "CustoPorCargo.csv"
    if not data_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig", decimal=",")
    except Exception:
        df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _get_custo_medio_por_grupo(custo_df: pd.DataFrame) -> Dict[str, float]:
    if custo_df is None or custo_df.empty:
        return {}
    col_cargo = None
    col_salario = None
    for col in custo_df.columns:
        col_norm = _norm_col_key(col)
        if col_norm == "cargo":
            col_cargo = col
        if "salario" in col_norm:
            col_salario = col
    if not col_cargo or not col_salario:
        return {}
    buckets = {"auxiliar": [], "lider": [], "aprend": []}
    for _, row in custo_df.iterrows():
        cargo = str(row.get(col_cargo, "")).strip()
        salario = row.get(col_salario)
        if salario is None or (isinstance(salario, float) and pd.isna(salario)):
            continue
        salario_val = safe_float(salario, float("nan"))
        if not math.isfinite(salario_val) or salario_val <= 0:
            continue
        cargo_norm = normalize_processo_nome(cargo)
        if "auxiliar" in cargo_norm:
            buckets["auxiliar"].append(float(salario_val))
        elif "lider" in cargo_norm:
            buckets["lider"].append(float(salario_val))
        elif "aprendiz" in cargo_norm:
            buckets["aprend"].append(float(salario_val))
    custo_medios: Dict[str, float] = {}
    for key, vals in buckets.items():
        if vals:
            custo_medios[key] = float(sum(vals) / len(vals))
    return custo_medios


def _calc_salario_map_from_sugestao(
    cargo_sugestao: Optional[Dict[str, float]],
    custo_medio: Dict[str, float],
) -> Tuple[Optional[float], Optional[Dict[str, object]]]:
    if not cargo_sugestao or not custo_medio:
        return None, None
    qtd_lider = int(round(float(cargo_sugestao.get("lideres", 0.0) or 0.0)))
    qtd_aux = int(round(float(cargo_sugestao.get("auxiliares", 0.0) or 0.0)))
    qtd_aprend = int(round(float(cargo_sugestao.get("aprend", 0.0) or 0.0)))
    total = 0.0
    detalhes = {
        "lideres": qtd_lider,
        "auxiliares": qtd_aux,
        "aprend": qtd_aprend,
        "custo_lider": custo_medio.get("lider"),
        "custo_aux": custo_medio.get("auxiliar"),
        "custo_aprend": custo_medio.get("aprend"),
    }
    for key, qtd in (("lider", qtd_lider), ("auxiliar", qtd_aux), ("aprend", qtd_aprend)):
        custo = custo_medio.get(key)
        if custo is not None and math.isfinite(custo):
            total += float(qtd) * float(custo)
    if total <= 0:
        return None, detalhes
    detalhes["salario_total"] = total
    return total, detalhes


def _normalize_iaf_value(raw_val: object) -> Optional[float]:
    iaf_val = safe_float(raw_val, float("nan"))
    if not math.isfinite(iaf_val) or iaf_val <= 0:
        return None
    if iaf_val > 1.5:
        iaf_val = iaf_val / 100.0
    return iaf_val if math.isfinite(iaf_val) and iaf_val > 0 else None


def _select_similar_lojas(
    train_df: pd.DataFrame,
    target_features: Dict[str, object],
    loja_nome: Optional[str],
    target_aux_hist: Optional[float] = None,
    n: int = 3,
) -> pd.DataFrame:
    if train_df is None or train_df.empty:
        return pd.DataFrame()
    feature_cols = [col for col in SIMILARITY_FEATURES if col in train_df.columns]
    if not feature_cols:
        return pd.DataFrame()
    df_feat = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = df_feat.median(skipna=True)
    df_feat = df_feat.fillna(medians)
    target_vals = []
    for col in feature_cols:
        val = safe_float(target_features.get(col), float("nan"))
        if not math.isfinite(val):
            val = float(medians.get(col, 0.0))
        target_vals.append(val)
    target_series = pd.Series(target_vals, index=feature_cols, dtype="float64")
    means = df_feat.mean()
    stds = df_feat.std().replace(0.0, 1.0)
    df_scaled = (df_feat - means) / stds
    target_scaled = ((target_series - means) / stds).to_numpy(dtype="float64")
    aux_weight = 3.0
    aux_series = pd.to_numeric(train_df.get("QtdAux"), errors="coerce")
    aux_median = float(aux_series.median(skipna=True)) if aux_series is not None else 0.0
    aux_series = aux_series.fillna(aux_median)
    aux_mean = float(aux_series.mean()) if aux_series is not None else 0.0
    aux_std = float(aux_series.std() or 1.0) if aux_series is not None else 1.0
    aux_std = aux_std if aux_std != 0.0 else 1.0
    aux_scaled = (aux_series - aux_mean) / aux_std
    target_aux = target_aux_hist
    if target_aux is None:
        target_aux = safe_float(target_features.get("QtdAux"), float("nan"))
    if not math.isfinite(target_aux):
        target_aux = aux_median
    target_aux_scaled = (float(target_aux) - aux_mean) / aux_std
    cluster_mask = None
    n_clusters = min(4, len(df_scaled))
    if n_clusters >= 2:
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(df_scaled)
            cluster_label = int(model.predict([target_scaled])[0])
            cluster_mask = labels == cluster_label
        except Exception:
            cluster_mask = None
    if cluster_mask is not None:
        df_candidates = train_df.loc[cluster_mask].copy()
        df_scaled_candidates = df_scaled.loc[cluster_mask]
        aux_scaled_candidates = aux_scaled.loc[cluster_mask]
    else:
        df_candidates = train_df.copy()
        df_scaled_candidates = df_scaled
        aux_scaled_candidates = aux_scaled
    if loja_nome and "Loja" in df_candidates.columns:
        df_candidates = df_candidates[df_candidates["Loja"].astype(str).str.strip().ne(loja_nome)]
        df_scaled_candidates = df_scaled_candidates.loc[df_candidates.index]
        aux_scaled_candidates = aux_scaled_candidates.loc[df_candidates.index]
    if df_candidates.empty:
        return pd.DataFrame()
    diffs = df_scaled_candidates.to_numpy(dtype="float64") - target_scaled
    dist_aux = aux_scaled_candidates.to_numpy(dtype="float64") - target_aux_scaled
    dist = np.sqrt((diffs ** 2).sum(axis=1) + (aux_weight * dist_aux) ** 2)
    df_candidates = df_candidates.assign(_sim_dist=dist)
    return df_candidates.sort_values("_sim_dist").head(n)


def render_calc_tab(tab_calc: DeltaGenerator) -> Dict[str, object]:
    """Renderiza a aba de cálculo até a preparação das features."""
    with tab_calc:
        # =============================================================================
        # Modo de calculo
        # =============================================================================
        st.subheader("Modo de cálculo")
        opcoes = [
            "Machine Learning",
            "Simplificado (Simulações)",
        ]
        if "modo_calc" not in st.session_state:
            st.session_state.modo_calc = opcoes[0]

        def set_modo(modo):
            st.session_state.modo_calc = modo

        cols = st.columns(2)
        for col, opcao in zip(cols, opcoes):
            with col:
                st.button(
                    opcao,
                    key=f"btn_{opcao}",
                    type="primary" if st.session_state.modo_calc == opcao else "secondary",
                    use_container_width=True,
                    on_click=set_modo,
                    kwargs={"modo": opcao},
                )
        modo_calc = st.session_state.modo_calc
        modo_ml = modo_calc == "Machine Learning"
        modo_simplificado = modo_calc == "Simplificado (Simulações)"

        st.divider()

        total_refs = get_total_reference_values(st.session_state.get("fIndicadores"))
        total_base_ativa_ref = total_refs.get("BaseAtivaTotal", 0.0)
        total_receita_ref = total_refs.get("ReceitaTotalMes", 0.0)

        # =============================================================================
        # Pesquisa de loja
        # =============================================================================
        st.markdown("**Pesquisar loja existente (opcional)**")
        col_lookup = st.columns([1, 1, 1])
        def _trigger_lookup_enter():
            st.session_state["lookup_enter_trigger"] = True
        with col_lookup[0]:
            lookup_field = st.radio(
                "Campo de pesquisa",
                ["Loja", "SAP", "BCPS"],
                horizontal=True,
                key="lookup_field",
                label_visibility="collapsed",
            )

        df_ind = st.session_state.get("fIndicadores")
        df_estrutura = st.session_state.get("dEstrutura")
        df_pessoas = st.session_state.get("dPessoas")
        data_version = st.session_state.get("_data_version")
        if (
            st.session_state.get("lookup_found")
            and st.session_state.get("lookup_loja_nome")
            and data_version
            and st.session_state.get("_lookup_data_version") != data_version
        ):
            loja_refresh = str(st.session_state.get("lookup_loja_nome", "")).strip()
            indicator_row, _ = _get_loja_row(df_ind, loja_refresh) if df_ind is not None else ({}, False)
            estrutura_row, _ = _get_loja_row(df_estrutura, loja_refresh) if df_estrutura is not None else ({}, False)
            pessoas_row, _ = _get_loja_row(df_pessoas, loja_refresh) if df_pessoas is not None else ({}, False)
            combined: Dict[str, object] = {}
            if estrutura_row:
                combined.update(estrutura_row)
            if indicator_row:
                combined.update(indicator_row)
            if pessoas_row:
                combined.update(pessoas_row)
            if combined:
                st.session_state["lookup_row"] = _standardize_row(combined)
                st.session_state["_lookup_data_version"] = data_version
        if lookup_field in ("BCPS", "SAP"):
            with col_lookup[1]:
                lookup_code = st.text_input(
                    "Codigo de pesquisa",
                    placeholder=(f"Código ({lookup_field})"),
                    key="lookup_code",
                    label_visibility="collapsed",
                    on_change=_trigger_lookup_enter,
                )
        elif lookup_field == "Loja":
            with col_lookup[1]:
                _ = st.text_input(
                    "Nome da loja (pesquisa)",
                    placeholder="Nome da loja",
                    key="lookup_loja_input",
                    label_visibility="collapsed",
                    on_change=_trigger_lookup_enter,
                )
            lookup_code = st.session_state.get("lookup_loja_input", "")

        with col_lookup[2]:
            lookup_submit = st.button("Pesquisar", use_container_width=True)
        lookup_trigger = bool(lookup_submit or st.session_state.pop("lookup_enter_trigger", False))
        if lookup_trigger:
            if ((df_ind is None or df_ind.empty) and (df_estrutura is None or df_estrutura.empty)):
                st.warning("⚠️ Bases de indicadores e de estrutura não estão carregadas.")
            elif not lookup_field or not lookup_code:
                st.warning("⚠️ Informe o campo e o valor para pesquisar.")
            else:
                colname = lookup_field
                code_norm = _norm_code(lookup_code)

                matches = pd.DataFrame()
                proceed = True
                if df_ind is not None and not df_ind.empty:
                    if colname not in df_ind.columns:
                        st.warning(f"⚠️ Coluna '{colname}' não encontrada na base de indicadores.")
                        if colname in ("BCPS", "SAP"):
                            proceed = False
                    series_norm = df_ind[colname].map(_norm_code)
                    if colname != "Loja":
                        mask = series_norm == code_norm
                    else:
                        mask = series_norm.str.contains(code_norm, na=False)
                    matches = df_ind.loc[mask]
                else:
                    if colname in ("BCPS", "SAP"):
                        st.warning("⚠️ Base de indicadores não disponível para pesquisa por BCPS/SAP.")
                        proceed = False

                if not proceed and (df_estrutura is None or df_estrutura.empty):
                    st.session_state["lookup_found"] = False
                    st.session_state["lookup_row"] = None
                else:
                    indicator_row = matches.iloc[0].to_dict() if not matches.empty else {}
                    estrutura_row: Dict[str, object] = {}
                    estrutura_used = False
                    pessoas_row: Dict[str, object] = {}
                    pessoas_used = False
                    if df_estrutura is not None and not df_estrutura.empty:
                        nome_ref = indicator_row.get("Loja") if indicator_row else lookup_code
                        estrutura_row, estrutura_ok = _get_loja_row(df_estrutura, nome_ref)
                        if not estrutura_ok and lookup_field == "Loja":
                            estrutura_row, estrutura_ok = _get_loja_row(df_estrutura, lookup_code)
                        if estrutura_ok:
                            for key, val in estrutura_row.items():
                                if key == "Loja":
                                    continue
                                if val not in (None, "") and not (isinstance(val, float) and pd.isna(val)):
                                    estrutura_used = True
                                    break
                        if not estrutura_used:
                            estrutura_row = {}
                    if df_pessoas is not None and not df_pessoas.empty:
                        nome_ref = indicator_row.get("Loja") if indicator_row else lookup_code
                        pessoas_row, pessoas_ok = _get_loja_row(df_pessoas, nome_ref)
                        if not pessoas_ok and lookup_field == "Loja":
                            pessoas_row, pessoas_ok = _get_loja_row(df_pessoas, lookup_code)
                        pessoas_used = bool(pessoas_row)

                    combined: Dict[str, object] = {}
                    if estrutura_row:
                        combined.update(estrutura_row)
                    if indicator_row:
                        combined.update(indicator_row)
                    if pessoas_row:
                        combined.update(pessoas_row)

                    if combined:
                        st.session_state["lookup_found"] = True
                        st.session_state["lookup_row"] = _standardize_row(combined)
                        if data_version:
                            st.session_state["_lookup_data_version"] = data_version
                        st.session_state["absenteismo_input"] = _compute_absenteismo_prefill(st.session_state.get("lookup_row", {}))
                        apply_operacional_defaults_from_lookup(st.session_state["lookup_row"])
                        # Reseta frequencias simuladas ao trocar de loja para permitir novo prefill automatico.
                        loja_lookup_prev = st.session_state.get("lookup_loja_nome")
                        loja_lookup_now = str(combined.get("Loja", lookup_code)).strip()
                        if loja_lookup_prev and loja_lookup_prev != loja_lookup_now:
                            sim_freq = st.session_state.get("sim_processos_freq")
                            if isinstance(sim_freq, dict):
                                sim_freq.pop(normalize_processo_nome("Mudanca de planograma"), None)
                        st.session_state["lookup_loja_nome"] = loja_lookup_now
                        # Reinicializa indicadores de entrada ao carregar nova loja
                        st.session_state["indicadores_reset_payload"] = {
                            "base_ativa": safe_float(get_lookup(combined, "BaseAtiva"), 0.0),
                            "receita_total": safe_float(get_lookup(combined, "ReceitaTotalMes"), 0.0),
                            "inicios": safe_float(get_lookup(combined, "Inicios"), 0.0),
                            "reinicios": safe_float(get_lookup(combined, "Reinicios"), 0.0),
                            "recuperados": safe_float(get_lookup(combined, "Recuperados"), 0.0),
                            "i4a_i6": safe_float(get_lookup(combined, "I4aI6"), 0.0),
                        }
                        loja_nome = str(combined.get("Loja", lookup_code)).strip()
                        fontes = []
                        if indicator_row:
                            fontes.append("Indicadores")
                        if estrutura_used:
                            fontes.append("Estrutura")
                        detalhe = f" ({' + '.join(fontes)})" if fontes else ""
                        st.success(f"✅ Loja encontrada: **{loja_nome}**{detalhe}")
                    else:
                        st.session_state["lookup_found"] = False
                        st.session_state["lookup_row"] = None
                        st.warning("⚠️ Nenhuma loja encontrada com esse valor.")

        if st.session_state.get("lookup_found") and st.session_state.get("lookup_row"):
            loja_nome = str(st.session_state["lookup_row"].get("Loja", "")).strip()
            st.info(f"Usando indicadores da loja: **{loja_nome}**")

        # Informar dados de base
        if modo_calc:
            lookup_prefill = st.session_state.get("lookup_row") or {}
            absenteismo_prefill = _compute_absenteismo_prefill(lookup_prefill)
            if "absenteismo_input" not in st.session_state:
                st.session_state["absenteismo_input"] = absenteismo_prefill
            col1, col2, col3 = st.columns(3)
            with col1:
                horas_disp_input = st.number_input(
                    "Horas contratuais (h/sem)",
                    min_value=5.0,
                    value=44.0,
                    step=0.5,
                    format="%.1f",
                    help="Carga semanal prevista em contrato para cada auxiliar, antes de qualquer perda operacional.",
                )
                horas_disp = float(horas_disp_input)
                if horas_disp_input > 200:
                    horas_disp = horas_disp_input / 4.33
                    st.caption(f"Valor informado parece mensal. Convertido para {horas_disp:.1f} h/semana.")
                horas_loja_config_raw = safe_float(
                    st.session_state.get("horas_operacionais_form", st.session_state.get("horas_loja_config", 10.0)),
                    10.0,
                )
                horas_loja_config = horas_loja_config_raw
            with col2:
                absenteismo = st.number_input(
                    "Absenteísmo (0–1)",
                    min_value=0.00,
                    max_value=0.30,
                    step=0.01,
                    format="%.2f",
                    help="Percentual médio perdido com faltas, férias e treinamentos. Será abatido das horas contratuais.",
                    key="absenteismo_input",
                )
            folga_base = 0.15
            with col3:
                folga_operacional = st.number_input(
                    "Folga operacional (0–1)",
                    min_value=0.00,
                    max_value=0.50,
                    value=folga_base,
                    step=0.01,
                    format="%.2f",
                    help="Percentual único que cobre monotonia, picos/SLA e margem tática. Quanto maior, mais folga no dimensionamento.",
                )
            dias_operacionais_mes = safe_float(st.session_state.get("dias_operacionais_mes"), 0.0)
            dias_operacionais_semana = safe_float(
                st.session_state.get("dias_operacionais_semana", st.session_state.get("dias_operacionais_loja_form", 6.0)),
                6.0,
            )
            if dias_operacionais_mes > 0:
                dias_operacionais_semana = dias_operacionais_mes / float(WEEKS_PER_MONTH)
            dias_operacionais_semana = max(1.0, min(7.0, float(dias_operacionais_semana)))
            if horas_loja_config <= 24:
                horas_loja_config = horas_loja_config * dias_operacionais_semana

            st.session_state["horas_disp_semanais"] = horas_disp
            st.session_state["horas_loja_config"] = horas_loja_config
            st.session_state["dias_operacionais_semana"] = dias_operacionais_semana
            if dias_operacionais_mes <= 0:
                st.session_state["dias_operacionais_mes"] = float(dias_operacionais_semana) * float(WEEKS_PER_MONTH)
            st.session_state["folga_operacional"] = float(folga_operacional)
        else:
            horas_disp = 44.0
            horas_loja_config = float(st.session_state.get("horas_loja_config", 10.0))
            absenteismo = float(DEFAULT_ABSENTEISMO)
            folga_operacional = 0.15

        dias_operacionais_mes = safe_float(st.session_state.get("dias_operacionais_mes"), 0.0)
        dias_operacionais_semana = safe_float(
            st.session_state.get("dias_operacionais_semana", st.session_state.get("dias_operacionais_loja_form", 6.0)),
            6.0,
        )
        if dias_operacionais_mes > 0:
            dias_operacionais_semana = dias_operacionais_mes / float(WEEKS_PER_MONTH)
        dias_operacionais_semana = max(1.0, min(7.0, float(dias_operacionais_semana)))
        st.session_state["dias_operacionais_semana"] = dias_operacionais_semana
        if dias_operacionais_mes <= 0:
            st.session_state["dias_operacionais_mes"] = float(dias_operacionais_semana) * float(WEEKS_PER_MONTH)
        st.session_state["folga_operacional"] = float(folga_operacional)

        dias_operacionais_em_uso = int(round(float(st.session_state.get("dias_operacionais_semana", dias_operacionais_semana))))
        dias_operacionais_em_uso = max(1, min(7, dias_operacionais_em_uso))

        ocupacao_alvo = float(DEFAULT_OCUPACAO_ALVO)
        fator_monotonia = 1.0 + folga_operacional if modo_calc else 1.0 + folga_operacional
        margem = folga_operacional
        sla_buffer = folga_operacional

        # Dados da loja até features_input
        lookup_row = st.session_state.get("lookup_row")
        has_lookup = isinstance(lookup_row, dict) and len(lookup_row) > 0

        with st.container():
            st.subheader("Dados da loja")
            st.markdown("**Estrutura Física**")
            estrutura_defaults: Dict[str, float] = {}
            estrutura_flags: Dict[str, bool] = {}
            if has_lookup:
                for key in ["Area Total", "Qtd Caixas", "HorasOperacionais", "DiasOperacionaisMes"]:
                    val = safe_float(get_lookup(lookup_row, key), 0.0)
                    if not pd.isna(val) and val is not None and val != 0.0:
                        estrutura_defaults[key] = val
                for key, col in [("Escritorio", "Escritorio"), ("Copa", "Copa"), ("Espaco Evento", "Espaco Evento"), ("Espaco Evento", "Esp Conv")]:
                    val = get_lookup(lookup_row, col)
                    if isinstance(val, str):
                        estrutura_flags[key] = val.strip().upper() in ("SIM", "VERDADEIRO", "TRUE", "1")
                    else:
                        estrutura_flags[key] = bool(val)
            area_total = float(estrutura_defaults.get("Area Total", 0.0) or 0.0)
            dias_operacionais_base = safe_float(st.session_state.get("dias_operacionais_mes"), 0.0)
            if dias_operacionais_base <= 0:
                dias_operacionais_base = float(st.session_state.get("dias_operacionais_semana", 6.0)) * float(WEEKS_PER_MONTH)
            colA, colB, colC = st.columns(3)
            with colA:
                qtd_caixas = st.number_input(
                    "Qtd Caixas",
                    min_value=0,
                    step=1,
                    value=int(estrutura_defaults.get("Qtd Caixas", 0.0)),
                )
                espaco_evento = st.selectbox(
                    "Espaco Evento",
                    ["Não", "Sim"],
                    index=1 if estrutura_flags.get("Espaco Evento") else 0,
                ) == "Sim"
            with colB:
                escritorio = st.selectbox(
                    "Escritorio",
                    ["Não", "Sim"],
                    index=1 if estrutura_flags.get("Escritorio") else 0,
                ) == "Sim"
                copa = st.selectbox(
                    "Copa",
                    ["Não", "Sim"],
                    index=1 if estrutura_flags.get("Copa") else 0,
                ) == "Sim"
            with colC:
                dias_operacionais_prefill = safe_float(
                    estrutura_defaults.get("DiasOperacionaisMes", dias_operacionais_base),
                    dias_operacionais_base,
                )
                dias_operacionais_prefill = max(1.0, min(31.0, dias_operacionais_prefill))
                dias_operacionais_loja = st.number_input(
                    "Dias operacionais ao mes",
                    min_value=1.0,
                    max_value=31.0,
                    step=0.1,
                    value=float(dias_operacionais_prefill),
                    format="%.1f",
                    help="Média histórica de dias em que a loja opera no mês.",
                )
                dias_operacionais_mes = float(dias_operacionais_loja)
                dias_operacionais_semana = max(
                    1.0,
                    min(7.0, float(dias_operacionais_mes) / float(WEEKS_PER_MONTH)),
                )
                dias_operacionais_em_uso = int(round(dias_operacionais_semana))
                dias_operacionais_em_uso = max(1, min(7, dias_operacionais_em_uso))
                horas_op_default = float(estrutura_defaults.get("HorasOperacionais", 0.0) or st.session_state.get("horas_loja_config", 0.0))
                if horas_op_default > 24 and dias_operacionais_semana > 0:
                    horas_op_default = horas_op_default / max(1.0, dias_operacionais_semana)
                horas_operacionais_input = st.number_input(
                    "Horas operacionais (h/dia)",
                    min_value=1.0,
                    max_value=24.0,
                    step=1.0,
                    value=float(horas_op_default),
                    format="%.1f",
                    help="Horas de funcionamento por dia. Alimenta os cálculos ideais/ML.",
                )
                horas_operacionais_diarias = float(horas_operacionais_input)
                horas_operacionais_semanais = horas_operacionais_diarias * max(1.0, dias_operacionais_semana)
            st.session_state["dias_operacionais_loja_form"] = dias_operacionais_em_uso
            st.session_state["dias_operacionais_semana"] = float(dias_operacionais_semana)
            st.session_state["dias_operacionais_mes"] = float(dias_operacionais_mes)
            st.session_state["horas_operacionais_form"] = float(horas_operacionais_diarias)
            st.session_state["horas_loja_config"] = float(horas_operacionais_diarias)

            st.divider()

            st.markdown("**Indicadores**")
            lookup_row = st.session_state.get("lookup_row")
            loja_nome_alvo = ""
            if has_lookup:
                loja_nome_alvo = str(lookup_row.get("Loja", "")).strip()
            else:
                loja_nome_alvo = str(st.session_state.get("lookup_loja_input", "")).strip()

            if has_lookup:
                base_ativa_val = get_lookup_value("BaseAtiva")
                receita_total_val = get_lookup_value("ReceitaTotalMes")
                inicios_val = get_lookup_value("Inicios")
                reinicios_val = get_lookup_value("Reinicios")
                recuperados_val = get_lookup_value("Recuperados")
                i4a_i6_val = get_lookup_value("I4aI6")
            else:
                base_ativa_val = 0.0
                receita_total_val = 0.0
                inicios_val = 0.0
                reinicios_val = 0.0
                recuperados_val = 0.0
                i4a_i6_val = 0.0
            reset_payload = st.session_state.pop("indicadores_reset_payload", None)
            if reset_payload:
                st.session_state["input_base_ativa"] = reset_payload.get("base_ativa", base_ativa_val)
                st.session_state["input_receita_total"] = reset_payload.get("receita_total", receita_total_val)
                st.session_state["input_inicios"] = reset_payload.get("inicios", inicios_val)
                st.session_state["input_reinicios"] = reset_payload.get("reinicios", reinicios_val)
                st.session_state["input_recuperados"] = reset_payload.get("recuperados", recuperados_val)
                st.session_state["input_i4a_i6"] = reset_payload.get("i4a_i6", i4a_i6_val)
            else:
                # Define defaults apenas se ainda não houver estado (evita o warning de valor duplo)
                st.session_state.setdefault("input_base_ativa", base_ativa_val)
                st.session_state.setdefault("input_receita_total", receita_total_val)
                st.session_state.setdefault("input_inicios", inicios_val)
                st.session_state.setdefault("input_reinicios", reinicios_val)
                st.session_state.setdefault("input_recuperados", recuperados_val)
                st.session_state.setdefault("input_i4a_i6", i4a_i6_val)

            colIndA, colIndB, colIndC = st.columns(3)
            with colIndA:
                base_ativa = st.number_input(
                    "Base Ativa",
                    min_value=0.0,
                    step=1.0,
                    key="input_base_ativa",
                )
                receita_total = st.number_input(
                    "Receita Total / Mês (R$)",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                    key="input_receita_total",
                )
            cluster_targets = [
                "Pedidos/Hora",
                "Pedidos/Dia",
                "Itens/Pedido",
                "Faturamento/Hora",
                "%Retirada",
            ]
            with colIndB:
                recuperados = st.number_input(
                    "Recuperados",
                    min_value=0.0,
                    step=1.0,
                    key="input_recuperados",
                )
                i4_a_i6 = st.number_input(
                    "I4 a I6",
                    min_value=0.0,
                    step=1.0,
                    key="input_i4a_i6",
                )
            with colIndC:
                inicios = st.number_input(
                    "Inícios",
                    min_value=0.0,
                    step=1.0,
                    key="input_inicios",
                )
                reinicios = st.number_input(
                    "Reinícios",
                    min_value=0.0,
                    step=1.0,
                    key="input_reinicios",
                )

            manual_override_indicadores = False
            base_ativa_override = False
            receita_total_override = False
            if has_lookup:
                original_vals = [
                    base_ativa_val,
                    receita_total_val,
                    inicios_val,
                    reinicios_val,
                    recuperados_val,
                    i4a_i6_val,
                ]
                current_vals = [base_ativa, receita_total, inicios, reinicios, recuperados, i4_a_i6]
                manual_override_indicadores = any(
                    safe_float(cur, 0.0) != safe_float(orig, 0.0) for cur, orig in zip(current_vals, original_vals)
                )
                base_ativa_override = safe_float(base_ativa, 0.0) != safe_float(base_ativa_val, 0.0)
                receita_total_override = safe_float(receita_total, 0.0) != safe_float(receita_total_val, 0.0)

            base_ativa_obs = safe_float(base_ativa_val, base_ativa) if has_lookup else base_ativa
            receita_total_obs = safe_float(receita_total_val, receita_total) if has_lookup else receita_total
            inicios_obs = safe_float(inicios_val, inicios) if has_lookup else inicios
            reinicios_obs = safe_float(reinicios_val, reinicios) if has_lookup else reinicios
            recuperados_obs = safe_float(recuperados_val, recuperados) if has_lookup else recuperados
            i4a_i6_obs = safe_float(i4a_i6_val, i4_a_i6) if has_lookup else i4_a_i6

            indicadores_ctx_observados = preparar_indicadores_operacionais(
                base_ativa=base_ativa_obs,
                receita_total=receita_total_obs,
                inicios=inicios_obs,
                reinicios=reinicios_obs,
                recuperados=recuperados_obs,
                i4_a_i6=i4a_i6_obs,
                total_base_ref=total_base_ativa_ref,
                total_receita_ref=total_receita_ref,
                cluster_targets=cluster_targets,
                indicadores_df=st.session_state.get("fIndicadores"),
                lookup_row=lookup_row if has_lookup else None,
                has_lookup=has_lookup,
                prefer_manual=False,
            )
            indicadores_ctx_estimados = preparar_indicadores_operacionais(
                base_ativa=base_ativa,
                receita_total=receita_total,
                inicios=inicios,
                reinicios=reinicios,
                recuperados=recuperados,
                i4_a_i6=i4_a_i6,
                total_base_ref=total_base_ativa_ref,
                total_receita_ref=total_receita_ref,
                cluster_targets=cluster_targets,
                indicadores_df=st.session_state.get("fIndicadores"),
                lookup_row=lookup_row if has_lookup else None,
                has_lookup=has_lookup,
                prefer_manual=manual_override_indicadores,
            )
            pct_base_total = indicadores_ctx_estimados["pct_base_total"]
            pct_faturamento = indicadores_ctx_estimados["pct_faturamento"]
            pct_ativos = indicadores_ctx_estimados["pct_ativos"]
            taxa_inicios = indicadores_ctx_estimados["taxa_inicios"]
            taxa_reativacao = indicadores_ctx_estimados["taxa_reativacao"]
            taxa_reinicio = indicadores_ctx_estimados["taxa_reinicio"]
            pct_ativos_obs = indicadores_ctx_observados["pct_ativos"]
            taxa_inicios_obs = indicadores_ctx_observados["taxa_inicios"]
            taxa_reativacao_obs = indicadores_ctx_observados["taxa_reativacao"]
            cluster_values_observados = indicadores_ctx_observados["cluster_values"]
            cluster_values_estimados = indicadores_ctx_estimados["cluster_values"]
            cluster_result = indicadores_ctx_estimados["cluster_result"]
            cluster_used = indicadores_ctx_estimados["cluster_used"]
            fluxo_base_ativa_used = False
            fluxo_receita_used = False
            relacao_base_ativa = {}
            relacao_receita = {}
            pedidos_dia_ref = safe_float(cluster_values_observados.get("Pedidos/Dia"), 0.0)
            pedidos_dia_delta_total = 0.0
            if receita_total_override:
                relacao_receita = fit_receita_pedidos_dia(st.session_state.get("fIndicadores"))
                pedidos_dia_rel = estimate_pedidos_dia_from_receita(receita_total, relacao_receita)
                if pedidos_dia_rel is not None:
                    receita_ref = safe_float(receita_total_obs, 0.0)
                    elasticity = safe_float(relacao_receita.get("elasticity"), 0.0)
                    if receita_ref > 0 and pedidos_dia_ref > 0 and safe_float(receita_total, 0.0) != receita_ref:
                        ratio = safe_float(receita_total, 0.0) / receita_ref
                        if elasticity > 0:
                            pedidos_dia_rel = pedidos_dia_ref * (ratio ** elasticity)
                        else:
                            pedidos_dia_rel = pedidos_dia_ref * ratio
                    pedidos_dia_delta_total += pedidos_dia_rel - pedidos_dia_ref
                    relacao_receita = dict(relacao_receita)
                    relacao_receita["horas_loja"] = float(horas_operacionais_diarias)
                    fluxo_receita_used = True
                else:
                    st.warning("Nao foi possivel estimar Pedidos/Dia pela relacao ReceitaTotalMes->Pedidos/Dia.")
            if base_ativa_override:
                relacao_base_ativa = fit_base_ativa_pedidos_dia(st.session_state.get("fIndicadores"))
                pedidos_dia_rel = estimate_pedidos_dia_from_base_ativa(base_ativa, relacao_base_ativa)
                if pedidos_dia_rel is not None:
                    base_ref = safe_float(base_ativa_obs, 0.0)
                    elasticity = safe_float(relacao_base_ativa.get("elasticity"), 0.0)
                    if base_ref > 0 and pedidos_dia_ref > 0 and safe_float(base_ativa, 0.0) != base_ref:
                        ratio = safe_float(base_ativa, 0.0) / base_ref
                        if elasticity > 0:
                            pedidos_dia_rel = pedidos_dia_ref * (ratio ** elasticity)
                        else:
                            pedidos_dia_rel = pedidos_dia_ref * ratio
                    pedidos_dia_delta_total += pedidos_dia_rel - pedidos_dia_ref
                    relacao_base_ativa = dict(relacao_base_ativa)
                    relacao_base_ativa["horas_loja"] = float(horas_operacionais_diarias)
                    fluxo_base_ativa_used = True
                else:
                    st.warning("Nao foi possivel estimar Pedidos/Dia pela relacao BaseAtiva->Pedidos/Dia.")
            if receita_total_override or base_ativa_override:
                cluster_values_estimados["Pedidos/Dia"] = max(0.0, pedidos_dia_ref + pedidos_dia_delta_total)
            if has_lookup:
                cluster_values_estimados["Itens/Pedido"] = cluster_values_observados.get("Itens/Pedido", 0.0)
                cluster_values_estimados["%Retirada"] = cluster_values_observados.get("%Retirada", 0.0)
            for msg_type, msg_text in indicadores_ctx_estimados["messages"]:
                if msg_type == "warning":
                    st.warning(msg_text)
                else:
                    st.info(msg_text)

            with st.expander("Indicadores derivados (cálculo automático)"):
                colDer1, colDer2, colDer3 = st.columns(3)
                with colDer1:
                    st.metric("% da Base Ativa total", f"{pct_base_total:.2f}%")
                    st.metric("Taxa Inícios", f"{taxa_inicios:.2f}%")
                with colDer2:
                    st.metric("% Ativos", f"{pct_ativos:.2f}%")
                    st.metric("Taxa Reativação", f"{taxa_reativacao:.2f}%")
                with colDer3:
                    st.metric("% do Faturamento Total", f"{pct_faturamento:.2f}%")
                    st.metric("Taxa Reinício", f"{taxa_reinicio:.2f}%")

            def _render_fluxos_expander(
                title: str,
                values: Dict[str, float],
                *,
                cluster_used: bool = False,
                cluster_result: Optional[Dict[str, object]] = None,
                label_overrides: Optional[Dict[str, str]] = None,
                deltas: Optional[Dict[str, float]] = None,
            ) -> None:
                label_overrides = label_overrides or {}
                deltas = deltas or {}
                with st.expander(title):
                    colFlow1, colFlow2, colFlow3 = st.columns(3)
                    with colFlow1:
                        val_hora = values.get("Pedidos/Hora", 0.0)
                        val_dia = values.get("Pedidos/Dia", 0.0)
                        label_hora = label_overrides.get("Pedidos/Hora", "Pedidos/Hora")
                        label_dia = label_overrides.get("Pedidos/Dia", "Pedidos/Dia")
                        delta_hora = deltas.get("Pedidos/Hora")
                        delta_dia = deltas.get("Pedidos/Dia")
                        delta_hora_disp = None
                        if delta_hora is not None and abs(delta_hora) > 0:
                            delta_hora_disp = f"{delta_hora:+.2f}"
                        delta_dia_disp = None
                        if delta_dia is not None and abs(delta_dia) > 0:
                            delta_dia_disp = f"{delta_dia:+.2f}"
                        st.metric(label_hora, f"{val_hora:.2f}", delta=delta_hora_disp)
                        st.metric(label_dia, f"{val_dia:.2f}", delta=delta_dia_disp)
                    with colFlow2:
                        delta_fat = deltas.get("Faturamento/Hora")
                        delta_fat_disp = None
                        if delta_fat is not None and abs(delta_fat) > 0:
                            delta_fat_disp = f"{delta_fat:+.2f}"
                        st.metric("Itens/Pedido", f"{values.get('Itens/Pedido', 0.0):.2f}")
                        st.metric(
                            "Faturamento/Hora",
                            f"R$ {values.get('Faturamento/Hora', 0.0):.2f}",
                            delta=delta_fat_disp,
                        )
                    with colFlow3:
                        st.metric("% Retirada", f"{values.get('%Retirada', 0.0):.2f}%")

            # Derivar faturamento/hora a partir de ReceitaTotalMes / (dias operacionais do mes * horas operacionais)
            dias_operacionais_mes = safe_float(st.session_state.get("dias_operacionais_mes"), 0.0)
            horas_operacionais_ref = float(horas_operacionais_diarias)
            if has_lookup:
                if dias_operacionais_mes <= 0:
                    dias_operacionais_mes = safe_float(get_lookup(lookup_row, "DiasOperacionaisMes"), 0.0)
                if horas_operacionais_ref <= 0:
                    horas_operacionais_ref = safe_float(get_lookup(lookup_row, "HorasOperacionais"), 0.0)
            if horas_operacionais_ref > 24 and dias_operacionais_semana > 0:
                horas_operacionais_ref = horas_operacionais_ref / max(1.0, float(dias_operacionais_semana))
            if horas_operacionais_ref <= 0:
                horas_operacionais_ref = float(horas_operacionais_diarias)
            if dias_operacionais_mes <= 0 and df_estrutura is not None and not df_estrutura.empty:
                if "DiasOperacionaisMes" in df_estrutura.columns:
                    dias_series = pd.to_numeric(df_estrutura["DiasOperacionaisMes"], errors="coerce")
                    dias_series = dias_series[dias_series > 0]
                    if not dias_series.empty:
                        dias_operacionais_mes = float(dias_series.mean())
            denom_fat_hora = max(0.1, float(horas_operacionais_ref)) * max(1.0, float(dias_operacionais_mes))

            def _apply_faturamento_hora(values: Dict[str, float], receita_ref: float) -> float:
                faturamento_val = values.get("Faturamento/Hora", 0.0)
                if receita_ref > 0 and denom_fat_hora > 0:
                    faturamento_hora_calc = float(receita_ref) / denom_fat_hora
                    if faturamento_hora_calc > 0:
                        faturamento_val = faturamento_hora_calc
                        values["Faturamento/Hora"] = faturamento_hora_calc
                return faturamento_val

            receita_ref_estimado = receita_total_obs if (has_lookup and not receita_total_override) else receita_total
            faturamento_hora_obs = cluster_values_observados.get("Faturamento/Hora", 0.0)
            if safe_float(faturamento_hora_obs, 0.0) <= 0:
                faturamento_hora_obs = _apply_faturamento_hora(cluster_values_observados, receita_total_obs)
            faturamento_hora_est = _apply_faturamento_hora(cluster_values_estimados, receita_ref_estimado)
            faturamento_hora = faturamento_hora_obs if (has_lookup and not receita_total_override) else faturamento_hora_est
            cluster_values_estimados["Faturamento/Hora"] = faturamento_hora

            op_caption = "Indicadores operacionais (dados historicos da loja selecionada)"
            if has_lookup and manual_override_indicadores:
                op_caption = "Indicadores operacionais (dados historicos da loja selecionada + mudanças manuais)"
            elif not has_lookup:
                op_caption = "Indicadores operacionais estimados por clusterizacao"
            values_to_render = dict(cluster_values_estimados)
            deltas_disp = {}
            label_overrides: Dict[str, str] = {}
            horas_ref = max(1.0, float(horas_operacionais_diarias))
            pedidos_dia_obs_val = safe_float(cluster_values_observados.get("Pedidos/Dia"), 0.0)
            pedidos_dia_est_val = safe_float(cluster_values_estimados.get("Pedidos/Dia"), 0.0)
            cluster_values_observados["Pedidos/Hora"] = pedidos_dia_obs_val / horas_ref if pedidos_dia_obs_val > 0 else 0.0
            cluster_values_estimados["Pedidos/Hora"] = pedidos_dia_est_val / horas_ref if pedidos_dia_est_val > 0 else 0.0
            if has_lookup:
                deltas_disp["Pedidos/Dia"] = cluster_values_estimados.get("Pedidos/Dia", 0.0) - cluster_values_observados.get("Pedidos/Dia", 0.0)
                deltas_disp["Pedidos/Hora"] = cluster_values_estimados.get("Pedidos/Hora", 0.0) - cluster_values_observados.get("Pedidos/Hora", 0.0)
                deltas_disp["Faturamento/Hora"] = cluster_values_estimados.get("Faturamento/Hora", 0.0) - cluster_values_observados.get("Faturamento/Hora", 0.0)
            if has_lookup and receita_total_override and base_ativa_override:
                label_overrides["Pedidos/Dia"] = "Pedidos/Dia (estimado por Receita + BaseAtiva)"
                label_overrides["Pedidos/Hora"] = "Pedidos/Hora (estimado por Receita + BaseAtiva)"
            elif has_lookup and receita_total_override:
                label_overrides["Pedidos/Dia"] = "Pedidos/Dia (estimado por Receita)"
                label_overrides["Pedidos/Hora"] = "Pedidos/Hora (estimado por Receita)"
            elif has_lookup and base_ativa_override:
                label_overrides["Pedidos/Dia"] = "Pedidos/Dia (estimado por BaseAtiva)"
                label_overrides["Pedidos/Hora"] = "Pedidos/Hora (estimado por BaseAtiva)"
            _render_fluxos_expander(
                op_caption,
                values_to_render,
                cluster_used=cluster_used,
                cluster_result=cluster_result,
                label_overrides=label_overrides,
                deltas=deltas_disp,
            )

            cluster_values = cluster_values_estimados
            pedidos_hora = cluster_values["Pedidos/Hora"]
            pedidos_dia = cluster_values["Pedidos/Dia"]
            itens_pedido = cluster_values["Itens/Pedido"]
            pct_retirada = cluster_values["%Retirada"]
            pedidos_hora_obs = cluster_values_observados["Pedidos/Hora"]
            pedidos_dia_obs = cluster_values_observados["Pedidos/Dia"]
            itens_pedido_obs = cluster_values_observados["Itens/Pedido"]
            pct_retirada_obs = cluster_values_observados["%Retirada"]

            cluster_values["Faturamento/Hora"] = cluster_values_estimados["Faturamento/Hora"]

            st.session_state["horas_operacionais_form"] = float(horas_operacionais_diarias)

            features_input_ideal = montar_features_input(
                area_total,
                qtd_caixas,
                float(horas_operacionais_diarias),
                float(dias_operacionais_em_uso),
                int(escritorio),
                int(copa),
                int(espaco_evento),
                base_ativa,
                receita_total,
                pct_ativos,
                taxa_inicios,
                taxa_reativacao,
                pedidos_hora,
                pedidos_dia,
                itens_pedido,
                faturamento_hora,
                pct_retirada,
            )
            features_input_hist = montar_features_input(
                area_total,
                qtd_caixas,
                float(horas_operacionais_diarias),
                float(dias_operacionais_em_uso),
                int(escritorio),
                int(copa),
                int(espaco_evento),
                base_ativa_obs,
                receita_total_obs,
                pct_ativos_obs,
                taxa_inicios_obs,
                taxa_reativacao_obs,
                pedidos_hora_obs,
                pedidos_dia_obs,
                itens_pedido_obs,
                faturamento_hora_obs,
                pct_retirada_obs,
            )
        with st.form("form_inputs"):
            st.markdown(
                """
                <style>
                div[data-testid="stForm"] {
                    border: none !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                }
                #calc-checkbox-group .st-emotion-cache-ocqkz7.e1f1d6gn5 {
                    align-items: center !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            # -----------------------------
            # Dados Manuseáveis (modo Simplificado)
            # -----------------------------
            if modo_simplificado:
                st.divider()
                st.markdown("### Dados manuseáveis (simulação)")

                st.caption(
                    "Os valores abaixo vêm **sugeridos** do histórico/clusterização quando possível, "
                    "mas servem apenas como ponto de partida. "
                    "Altere livremente para simular contextos de demanda e operação."
                )

                tmedio_min_atend = float(st.session_state.get("tmedio_min_atend", 6.0))
                sim_col1, sim_col2, sim_col3 = st.columns(3)
                with sim_col1:
                    sim_pedidos_dia = st.number_input(
                        "Pedidos/Dia (simulação)",
                        min_value=0.0,
                        step=1.0,
                        value=cluster_values["Pedidos/Dia"] if (cluster_used or has_lookup) else 0.0,
                        format="%.0f",
                        key="sim_pedidos_dia",
                        help="Volume total processado em um dia típico da loja. Use dados reais ou o cenário a testar.",
                    )
                    sim_itens_pedido = st.number_input(
                        "Itens por pedido (simulação)",
                        min_value=0.0,
                        step=0.1,
                        value=cluster_values["Itens/Pedido"] if (cluster_used or has_lookup) else 0.0,
                        format="%.1f",
                        key="sim_itens_pedido",
                        help="Quantidade média de itens manipulados a cada pedido. Afeta o esforço por atendimento.",
                    )
                with sim_col2:
                    tmedio_min_atend = st.number_input(
                        "Tempo médio de atendimento",
                        min_value=0.0,
                        step=0.5,
                        value=10.0,
                        format="%.1f",
                        key="tmedio_min_atend",
                        help="Tempo efetivo gasto para liberar um pedido completo (da chegada à entrega).",
                    )
                    sim_pct_retirada = st.number_input(
                        "% Retirada (simulação)",
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        value=cluster_values["%Retirada"] if (cluster_used or has_lookup) else 0.0,
                        format="%.1f",
                        key="sim_pct_retirada",
                        help="Vendas em caixa também são consideradas retirada e devem ser consideradas.",
                    )
                with sim_col3:
                    sim_faturamento_hora = st.number_input(
                        "Faturamento/Hora (simulação)",
                        min_value=0.0,
                        step=0.1,
                        value=cluster_values["Faturamento/Hora"] if (cluster_used or has_lookup) else 0.0,
                        format="%.2f",
                        key="sim_faturamento_hora",
                        help="Opcional. Use quando quiser alinhar a demanda via faturamento médio/hora; se 0, o sistema tenta inferir.",
                    )

                sim_payload = {
                    "pedidos_dia": sim_pedidos_dia,
                    "faturamento_hora": sim_faturamento_hora,
                    "itens_pedido": sim_itens_pedido,
                    "tmedio_min_atend": tmedio_min_atend,
                    "pct_retirada": sim_pct_retirada,
                }
                st.session_state["sim_inputs"] = sim_payload
                st.session_state["dados_manuseaveis"] = sim_payload.copy()

                tempo_global_dict, tempo_loja_dict = preparar_dicionarios_tempos_processos(
                    st.session_state.get("dAmostras"),
                    loja_nome_alvo,
                )
                st.session_state["sim_processos_tempos_global"] = tempo_global_dict
                st.session_state["sim_processos_tempos_loja"] = tempo_loja_dict

                # Frequencias no modo simplificado: so planograma tem prefill automatico.
                auto_freqs: Dict[str, float] = {}
                auto_freq_notes: Dict[str, str] = {}
                planograma_key = normalize_processo_nome("Mudanca de planograma")
                if has_lookup:
                    prateleiras_lookup = _get_prateleiras_lookup(lookup_prefill)
                    if prateleiras_lookup > 0:
                        auto_freqs[planograma_key] = float(round(prateleiras_lookup))
                        auto_freq_notes[planograma_key] = "Valor puxado da base da loja (Qtd Prateleiras)."
                    else:
                        prateleiras_est = _estimate_prateleiras_from_area(area_total, df_estrutura)
                        if prateleiras_est is not None and prateleiras_est > 0:
                            auto_freqs[planograma_key] = float(round(prateleiras_est))
                            auto_freq_notes[planograma_key] = (
                                "Estimado por regressao (Area Total x Qtd Prateleiras) usando dEstrutura."
                            )
                sim_processos_freq_state = st.session_state.get("sim_processos_freq", {}) or {}
                sim_processos_tempo_state = st.session_state.get("sim_processos_tempos_custom", {}) or {}
                with st.expander("Processos complementares (tempos e frequências)"):
                    st.caption(
                        "Tempos médios puxados de dAmostras (loja ou média geral), mas editáveis para simulação. "
                        "Frequências em ocorrências por semana."
                    )
                    updated_freqs: Dict[str, float] = {}
                    updated_tempos: Dict[str, float] = {}
                    for proc in PROCESSOS_PRIORITARIOS:
                        proc_norm = normalize_processo_nome(proc)
                        tempo_default = sim_processos_tempo_state.get(proc_norm)
                        usa_media_geral = proc_norm not in tempo_loja_dict and proc_norm in tempo_global_dict
                        if tempo_default is None or tempo_default < 0:
                            tempo_default = tempo_loja_dict.get(proc_norm) or tempo_global_dict.get(proc_norm) or 0.0
                        freq_default = sim_processos_freq_state.get(proc_norm)
                        if freq_default is None or freq_default <= 0:
                            freq_default = auto_freqs.get(proc_norm, 0.0)
                        with st.container():
                            fallback_label = " _(média geral)_" if usa_media_geral else ""
                            st.markdown(f"**{proc.strip()}**{fallback_label}")
                            col_tempo, col_freq = st.columns(2)
                            with col_tempo:
                                tempo_val = st.number_input(
                                    "Tempo médio (min)",
                                    min_value=0.0,
                                    step=0.5,
                                    value=float(tempo_default),
                                    format="%.2f",
                                    key=f"sim_proc_tempo_{proc_norm}",
                                )
                            with col_freq:
                                freq_help = None
                                if proc_norm == planograma_key:
                                    freq_help = "Para Mudanca de planograma, a frequencia e a quantidade de prateleiras da loja."
                                freq_val = st.number_input(
                                    "Freq/semana",
                                    min_value=0.0,
                                    step=1.0,
                                    value=float(freq_default),
                                    format="%.1f",
                                    key=f"sim_proc_freq_{proc_norm}",
                                    help=freq_help,
                                )
                            if usa_media_geral:
                                st.caption("Tempo vindo da média geral das lojas (sem dado específico da loja).")
                            if proc_norm == planograma_key:
                                st.caption("Mudanca de planograma: frequencia = quantidade de prateleiras da loja.")
                                auto_note = auto_freq_notes.get(proc_norm)
                                if auto_note and (proc_norm not in sim_processos_freq_state or sim_processos_freq_state.get(proc_norm, 0) <= 0):
                                    st.caption(f"Prefill automatico aplicado: {auto_note}")
                        updated_tempos[proc_norm] = tempo_val
                        updated_freqs[proc_norm] = freq_val
                st.session_state["sim_processos_tempos_custom"] = updated_tempos
                st.session_state["sim_processos_freq"] = updated_freqs
                st.session_state["sim_processos_auto_freq"] = auto_freqs
 
            criterio_options = get_criterio_mapeado_options()
            criterio_default = get_criterio_mapeado_label()
            criterio_index = criterio_options.index(criterio_default) if criterio_default in criterio_options else 0
            criterio_label = st.selectbox(
                "Critério de referência para estimativa ideal",
                options=criterio_options,
                index=criterio_index,
                key="criterio_mapeado_label_calc",
                help=MAPEADO_HELPER_TEXT,
            )
            st.session_state["criterio_mapeado_label"] = criterio_label
            criterio_key = get_criterio_mapeado_key()

            mostrar_metricas = False
            mostrar_sugestoes_cargo = False
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("<div id='calc-checkbox-group'>", unsafe_allow_html=True)
                if modo_ml:
                    mostrar_metricas = st.checkbox(
                        "Mostrar métricas/IC",
                        value=False,
                        help="Ative apenas se precisar das métricas e intervalos de confiança. Mantendo desativado o cálculo fica mais rápido.",
                    )
                mostrar_sugestoes_cargo = st.checkbox(
                    "Retornar sugestões por cargo",
                    value=False,
                    help="Mostra a divisao sugerida entre Lideres, Auxiliares e Aprend abaixo do card Ideal.",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:

                if modo_ml:

                    anchor_options = [50 + (2.5 * i) for i in range(17)]
                    raw_anchor = st.session_state.get("anchor_rpa_percent", 60)
                    try:
                        raw_anchor = float(raw_anchor)
                    except Exception:
                        raw_anchor = 60.0
                    if raw_anchor not in anchor_options:
                        raw_anchor = min(anchor_options, key=lambda v: abs(v - raw_anchor))

                    anchor_percent = st.select_slider(

                        f"Âncora {criterio_label} (%)",

                        options=anchor_options,

                        value=raw_anchor,

                        help=(
                            f"Percentil de {criterio_label} usado como referência: se a meta é evitar falta de gente, "
                            "prefira percentil mais baixo; se a meta é eficiência agressiva, prefira percentil mais alto."
                        ),

                    )

                else:

                    anchor_percent = float(st.session_state.get("anchor_rpa_percent", 60))

        with col3:
            st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Calcular Time Comercial",
                    type="primary",
                    use_container_width=True,
                )

    dias_operacionais_ativos = int(st.session_state.get("dias_operacionais_semana", dias_operacionais_semana))
    dias_operacionais_ativos = max(1, min(7, dias_operacionais_ativos))

    if submitted:
        st.session_state["anchor_rpa_percent"] = anchor_percent
        st.session_state["anchor_rpa_quantile"] = float(anchor_percent) / 100.0
    anchor_quantile = float(st.session_state.get("anchor_rpa_quantile", float(anchor_percent) / 100.0))

    if not submitted:
        return

    campos_obrigatorios = [
        area_total,
        base_ativa,
        receita_total,
        pedidos_dia,
        faturamento_hora,
    ]
    if not any(val and val > 0 for val in campos_obrigatorios):
        st.warning("Preencha os dados da loja (base/receita/pedidos/faturamento) antes de calcular.")
        return

    loja_nome_alvo_submit = None
    if st.session_state.get("lookup_found") and st.session_state.get("lookup_row"):
        loja_nome_alvo_submit = str(st.session_state["lookup_row"].get("Loja", "")).strip() or None

    train_df = prepare_training_dataframe(
        st.session_state["dEstrutura"],
        st.session_state["dPessoas"],
        st.session_state["fIndicadores"],
    )
    train_df = clean_training_dataframe(train_df)
    if train_df.empty:
        st.error("Sem dados válidos para treinar os modelos. Verifique dEstrutura/dPessoas/fIndicadores.")
        return
    elif len(train_df) < 15:
        st.info(f"A base de treino possui apenas {len(train_df)} lojas. As métricas podem variar bastante.")

    estoq_train_df = prepare_estoquistas_training_dataframe(
        st.session_state.get("dEstrutura"),
        st.session_state.get("dPessoas"),
        st.session_state.get("fIndicadores"),
    )

    model_bundle_hist = None
    model_bundle_ideal = None
    if modo_ml:
        criterio_hash = zlib.adler32(str(criterio_key).encode("utf-8")) % 10000
        cache_ver_hist = 9 + (CATBOOST_PARAM_VERSION * 1000)
        cache_ver_ideal = 9 + int(anchor_quantile * 100) + (CATBOOST_PARAM_VERSION * 1000) + criterio_hash
        model_bundle_hist = _train_cached(
            train_df,
            "historico",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver_hist,
        )
        def _warn_model_issue(bundle: Optional[Dict[str, object]], label: str) -> None:
            errors = (bundle or {}).get("errors") or {}
            msg = errors.get("catboost") or errors.get("_geral")
            if msg:
                st.warning(f"Modelo {label} indisponivel: {msg}")
        _warn_model_issue(model_bundle_hist, "historico")
        _warn_model_issue(model_bundle_ideal, "ideal")
        model_bundle_ideal = _train_cached(
            train_df,
            "ideal",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver_ideal,
        )

    estrutura_df = st.session_state.get("dEstrutura")
    pessoas_df = st.session_state.get("dPessoas")
    horas_por_colab = float(st.session_state.get("horas_disp_semanais", horas_disp))
    horas_loja_manual = float(st.session_state.get("horas_loja_config", horas_loja_config))
    manual_horas_form = safe_float(st.session_state.get("horas_operacionais_form"), 0.0)
    horas_loja, dias_operacionais_ativos = preparar_contexto_operacional(
        loja_nome_alvo_submit,
        estrutura_df,
        pessoas_df,
        manual_horas_form,
        dias_operacionais_em_uso,
        dias_operacionais_ativos,
        horas_loja_manual,
        dias_operacionais_mes=safe_float(st.session_state.get("dias_operacionais_mes"), 0.0),
    )
    tmedio_min_atend = float(st.session_state.get("tmedio_min_atend", 6.0))
    result_ideal = None
    features_input_hist_ml = features_input_hist
    features_input_ideal_ml = features_input_ideal
    indicator_feature_keys = [
        "BaseAtiva",
        "ReceitaTotalMes",
        "%Ativos",
        "TaxaInicios",
        "TaxaReativacao",
        "Pedidos/Hora",
        "Pedidos/Dia",
        "Itens/Pedido",
        "Faturamento/Hora",
        "%Retirada",
    ]
    beta_base_aux = estimate_elasticity_base_to_aux(
        train_df,
        horas_disp=horas_disp,
        margem=margem,
        anchor_quantile=anchor_quantile,
    )
    if modo_ml and loja_nome_alvo_submit:
        feature_row_ml, _ = _get_loja_row(train_df, loja_nome_alvo_submit)
        if feature_row_ml:
            features_input_hist_ml = feature_row_ml
            if not (manual_override_indicadores or base_ativa_override):
                features_input_ideal_ml = feature_row_ml
            else:
                features_input_ideal_ml = dict(feature_row_ml)
                for key in indicator_feature_keys:
                    if key in features_input_ideal:
                        features_input_ideal_ml[key] = features_input_ideal[key]

    if modo_ml:
        resultados_modelos: List[Dict[str, object]] = []
        resultados_modelos_ideal: List[Dict[str, object]] = []
        model_errors_hist: Dict[str, object] = {}
        model_errors_ideal: Dict[str, object] = {}
        global_importance_ideal: List[Dict[str, object]] = []
        skip_cap_cols_ideal = ["BaseAtiva", "Pedidos/Dia", "Pedidos/Hora"] if base_ativa_override else None
        if model_bundle_hist is not None:
            resultados_modelos, model_errors_hist = gerar_resultados_modelos(
                model_bundle_hist,
                train_df,
                features_input_hist_ml,
                "historico",
                horas_disp,
                margem,
                algo_order=["catboost"],
                anchor_quantile=anchor_quantile,
                apply_cluster_blend=False,
                compute_metrics=False,
            )
            resultados_modelos = [res for res in resultados_modelos if res.get("key") == "catboost"]
        if model_bundle_ideal is not None:
            resultados_modelos_ideal, model_errors_ideal = gerar_resultados_modelos(
                model_bundle_ideal,
                train_df,
                features_input_ideal_ml,
                "ideal",
                horas_disp,
                margem,
                algo_order=["catboost"],
                anchor_quantile=anchor_quantile,
                apply_cluster_blend=False,
                compute_metrics=mostrar_metricas,
                metrics_cache_bust=cache_ver_ideal,
                skip_cap_cols=skip_cap_cols_ideal,
            )
            resultados_modelos_ideal = [res for res in resultados_modelos_ideal if res.get("key") == "catboost"]
            model_cb_ideal = (model_bundle_ideal or {}).get("models", {}).get("catboost")
            import_rows = get_catboost_feature_importance(model_cb_ideal)
            if import_rows:
                global_importance_ideal = [
                    {"Feature": name, "Importancia": value} for name, value in import_rows
                ]

        cat_hist = resultados_modelos[0] if resultados_modelos else None
        cat_ideal = resultados_modelos_ideal[0] if resultados_modelos_ideal else None

        if not resultados_modelos:
            err_msgs = []
            for key, msg in (model_errors_hist or {}).items():
                label = MODEL_ALGO_NAMES.get(key, key) if key != "_geral" else "Modelo"
                err_msgs.append(f"{label}: {msg}")
            detalhe = "; ".join(err_msgs) if err_msgs else "Faca upload de dEstrutura, dPessoas e (opcional) fIndicadores."
            st.error(f"Nao ha modelos treinados (Historico). {detalhe}")

        if not resultados_modelos_ideal:
            err_msgs = []
            for key, msg in (model_errors_ideal or {}).items():
                label = MODEL_ALGO_NAMES.get(key, key) if key != "_geral" else "Modelo"
                err_msgs.append(f"{label}: {msg}")
            detalhe = "; ".join(err_msgs) if err_msgs else "Faca upload de dEstrutura, dPessoas e fIndicadores suficientes."
            st.error(f"Nao ha modelos treinados (Ideal). {detalhe}")

        ci_hist = {}
        ci_ideal = {}
        if mostrar_metricas and cat_hist:
            ci_hist = calcular_intervalos_modelos(
                train_df,
                features_input_hist_ml,
                "historico",
                horas_disp,
                margem,
                ["catboost"],
                anchor_quantile=anchor_quantile,
                apply_cluster_blend=False,
            ).get("catboost", {})
        if mostrar_metricas and cat_ideal:
            ci_ideal = calcular_intervalos_modelos(
                train_df,
                features_input_ideal_ml,
                "ideal",
                horas_disp,
                margem,
                ["catboost"],
                anchor_quantile=anchor_quantile,
                apply_cluster_blend=False,
                skip_cap_cols=skip_cap_cols_ideal,
            ).get("catboost", {})

        if cat_hist and cat_ideal:
            st.success("Previsao (Machine Learning) concluida!")
            if loja_nome_alvo_submit:
                st.markdown(
                    "<div style='height:1px;background:#e5e7eb;margin:12px 0 8px 0;'></div>"
                    f"<div style='text-align:center;font-size:1.4rem;font-weight:700;color:#0c0863;'>"
                    f"{loja_nome_alvo_submit}</div>",
                    unsafe_allow_html=True,
                )
            pred_hist_raw = float(cat_hist.get("pred") or 0.0)
            pred_ideal_raw = float(cat_ideal.get("pred") or 0.0)
            pred_ideal_model = pred_ideal_raw
            # Ajuste de elasticidade para manter monotonicidade em relação à BaseAtiva
            if base_ativa_override and beta_base_aux and base_ativa_obs > 0:
                ratio_base = safe_float(base_ativa, 0.0) / max(base_ativa_obs, 1e-6)
                ratio_flux = 1.0
                if pedidos_dia_obs > 0:
                    ratio_flux = safe_float(pedidos_dia, 0.0) / max(pedidos_dia_obs, 1e-6)
                beta_flux = 0.8
                fator_base = ratio_base ** beta_base_aux if ratio_base > 0 else 1.0
                fator_flux = ratio_flux ** beta_flux if ratio_flux > 0 else 1.0
                fator = fator_base * fator_flux
                base_pred = pred_ideal_raw
                if math.isfinite(fator):
                    if ratio_base < 1.0 or ratio_flux < 1.0:
                        pred_ideal_raw = min(base_pred, base_pred * fator)
                    elif ratio_base > 1.0 or ratio_flux > 1.0:
                        pred_ideal_raw = max(base_pred, base_pred * fator)
                    if ci_ideal:
                        for key_ci in ["ci_low", "ci_high", "ci_low_disp", "ci_high_disp", "ci_mid_disp", "pred_mean"]:
                            if key_ci in ci_ideal and ci_ideal[key_ci] is not None and math.isfinite(ci_ideal[key_ci]):
                                if ratio_base < 1.0 or ratio_flux < 1.0:
                                    ci_ideal[key_ci] = min(float(ci_ideal[key_ci]), float(ci_ideal[key_ci]) * fator)
                                else:
                                    ci_ideal[key_ci] = max(float(ci_ideal[key_ci]), float(ci_ideal[key_ci]) * fator)
                # Clampa contra o baseline da loja para evitar saltos contra-intuitivos
                pred_base_ref = None
                try:
                    features_base = dict(features_input_ideal_ml)
                    features_base["BaseAtiva"] = float(base_ativa_obs)
                    features_base["Pedidos/Dia"] = float(pedidos_dia_obs)
                    features_base["Pedidos/Hora"] = float(pedidos_hora_obs)
                    features_base["ReceitaTotalMes"] = float(receita_total_obs)
                    base_results, _ = gerar_resultados_modelos(
                        model_bundle_ideal,
                        train_df,
                        features_base,
                        "ideal",
                        horas_disp,
                        margem,
                        algo_order=["catboost"],
                        anchor_quantile=anchor_quantile,
                        apply_cluster_blend=False,
                        compute_metrics=False,
                        skip_cap_cols=skip_cap_cols_ideal,
                    )
                    if base_results:
                        pred_base_ref = float(base_results[0].get("pred") or 0.0)
                except Exception:
                    pred_base_ref = None
                if pred_base_ref is not None and math.isfinite(pred_base_ref):
                    if ratio_base < 1.0:
                        pred_ideal_raw = min(pred_ideal_raw, pred_base_ref)
                        if ci_ideal:
                            for key_ci in ["ci_low", "ci_high", "ci_low_disp", "ci_high_disp", "ci_mid_disp", "pred_mean"]:
                                if key_ci in ci_ideal and ci_ideal[key_ci] is not None and math.isfinite(ci_ideal[key_ci]):
                                    ci_ideal[key_ci] = min(float(ci_ideal[key_ci]), pred_base_ref)
                    elif ratio_base > 1.0:
                        pred_ideal_raw = max(pred_ideal_raw, pred_base_ref)
                        if ci_ideal:
                            for key_ci in ["ci_low", "ci_high", "ci_low_disp", "ci_high_disp", "ci_mid_disp", "pred_mean"]:
                                if key_ci in ci_ideal and ci_ideal[key_ci] is not None and math.isfinite(ci_ideal[key_ci]):
                                    ci_ideal[key_ci] = max(float(ci_ideal[key_ci]), pred_base_ref)
            if ci_ideal:
                ci_low_val = ci_ideal.get("ci_low")
                ci_high_val = ci_ideal.get("ci_high")
                if ci_low_val is not None and math.isfinite(ci_low_val):
                    ci_ideal["ci_low_disp"] = float(ci_low_val)
                if ci_high_val is not None and math.isfinite(ci_high_val):
                    ci_ideal["ci_high_disp"] = float(ci_high_val)
            pred_hist_int = int(round(pred_hist_raw))
            pred_ideal_int = int(round(pred_ideal_raw))
            diff_val = pred_ideal_raw - pred_hist_raw
            criterio_row = st.session_state.get("lookup_row") or {}
            criterio_is_iaf = criterio_key == "SalarioMapeadoIAF25"
            criterio_source_key = "SalarioMapeado" if criterio_is_iaf else criterio_key
            salario_map_ideal = None
            if criterio_key in ("SalarioMapeado", "SalarioMapeadoIAF25"):
                custo_df = _load_custo_por_cargo_df()
                custo_medio = _get_custo_medio_por_grupo(custo_df)
                cargo_sugestao_calc = _compute_cargo_suggestion(
                    pessoas_df,
                    st.session_state.get("fIndicadores"),
                    pred_ideal_raw,
                    base_ativa,
                    pedidos_dia,
                    area_total,
                    ASG_SEG_PEDIDOS_DIA_THRESHOLD,
                    ASG_SEG_AREA_TOTAL_THRESHOLD,
                )
                total_aprendizes = _resolve_total_colabs_for_aprendizes(
                    st.session_state.get("lookup_row"),
                    pessoas_df,
                    st.session_state.get("fIndicadores"),
                    base_ativa,
                    receita_total,
                )
                cargo_sugestao_calc, _ = _apply_aprendizes_rule(cargo_sugestao_calc, total_aprendizes)
                salario_map_ideal, _ = _calc_salario_map_from_sugestao(cargo_sugestao_calc, custo_medio)
            if criterio_key == "TotalMapeado":
                criterio_denom_hist = pred_hist_raw
                criterio_denom_ideal = pred_ideal_raw
            else:
                criterio_denom_hist = safe_float(criterio_row.get(criterio_source_key), float("nan"))
                if not math.isfinite(criterio_denom_hist) or criterio_denom_hist <= 0:
                    criterio_denom_hist = pred_hist_raw
                criterio_denom_ideal = (
                    float(salario_map_ideal)
                    if salario_map_ideal is not None and math.isfinite(salario_map_ideal) and salario_map_ideal > 0
                    else safe_float(criterio_row.get(criterio_source_key), float("nan"))
                )
                if not math.isfinite(criterio_denom_ideal) or criterio_denom_ideal <= 0:
                    criterio_denom_ideal = pred_ideal_raw
            receita_aux_hist = None
            if receita_total_obs > 0 and criterio_denom_hist > 0:
                receita_aux_hist = receita_total_obs / criterio_denom_hist
            receita_aux_ideal = None
            if receita_total > 0 and criterio_denom_ideal > 0:
                receita_aux_ideal = receita_total / criterio_denom_ideal
            if criterio_is_iaf:
                iaf_candidates = [
                    criterio_row.get("%IAF25"),
                    (features_input_ideal_ml or {}).get("%IAF25"),
                    (features_input_ideal or {}).get("%IAF25"),
                ]
                if loja_nome_alvo_submit:
                    iaf_from_train = (features_input_ideal_ml or {}).get("%IAF25")
                    if iaf_from_train is not None:
                        iaf_candidates.append(iaf_from_train)
                iaf_val = None
                for raw in iaf_candidates:
                    iaf_val = _normalize_iaf_value(raw)
                    if iaf_val is not None:
                        break
                if iaf_val is not None:
                    if receita_aux_hist is not None:
                        receita_aux_hist = receita_aux_hist * iaf_val
                    if receita_aux_ideal is not None:
                        receita_aux_ideal = receita_aux_ideal * iaf_val
            if criterio_key in ("SalarioMapeado", "SalarioMapeadoIAF25"):
                receita_aux_hist_disp = f"{receita_aux_hist:,.2f}" if receita_aux_hist else "-"
                receita_aux_ideal_disp = f"{receita_aux_ideal:,.2f}" if receita_aux_ideal else "-"
            else:
                receita_aux_hist_disp = f"R$ {receita_aux_hist:,.2f}" if receita_aux_hist else "-"
                receita_aux_ideal_disp = f"R$ {receita_aux_ideal:,.2f}" if receita_aux_ideal else "-"
            urgency_badge = _render_urgency_badge(diff_val)
            cargo_sugestao_html = ""
            seg_sugestao_html = ""
            if mostrar_sugestoes_cargo:
                estoq_feature_row: Dict[str, object] = {}
                lookup_row = st.session_state.get("lookup_row") or {}
                if isinstance(lookup_row, dict):
                    estoq_feature_row.update(lookup_row)
                estoq_feature_row.update(
                    {
                        "BaseAtiva": base_ativa,
                        "Pedidos/Dia": pedidos_dia,
                        "Pedidos/Hora": pedidos_hora,
                        "Itens/Pedido": itens_pedido,
                        "Faturamento/Hora": faturamento_hora,
                        "ReceitaTotalMes": receita_total,
                        "%Retirada": pct_retirada,
                    }
                )
                estoq_pred_raw, estoq_helper_text = _predict_estoquistas_sugestao(
                    estoq_train_df, estoq_feature_row
                )
                estoq_pred_int = int(round(estoq_pred_raw))
                estoq_helper_icon = _build_info_icon(estoq_helper_text)
                asg_sugerido = _suggest_staff_from_pedidos_dia(pedidos_dia, area_total)
                seg_sugerido = _suggest_staff_from_pedidos_dia(pedidos_dia, area_total)
                cargo_sugestao = _compute_cargo_suggestion(
                    pessoas_df,
                    st.session_state.get("fIndicadores"),
                    pred_ideal_raw,
                    base_ativa,
                    pedidos_dia,
                    area_total,
                    ASG_SEG_PEDIDOS_DIA_THRESHOLD,
                    ASG_SEG_AREA_TOTAL_THRESHOLD,
                )
                if cargo_sugestao:
                    total_aprendizes = _resolve_total_colabs_for_aprendizes(
                        st.session_state.get("lookup_row"),
                        pessoas_df,
                        st.session_state.get("fIndicadores"),
                        base_ativa,
                        receita_total,
                    )
                    cargo_sugestao, aprendiz_helper_text = _apply_aprendizes_rule(
                        cargo_sugestao, total_aprendizes
                    )
                    faixa_helper_text = (
                        f"Faixa BaseAtiva: {cargo_sugestao['faixa_label']} | "
                        f"Aux/Lider: {cargo_sugestao['aux_lider']:.2f}"
                    )
                    faixa_helper_icon = _build_info_icon(faixa_helper_text)
                    aprendiz_helper_icon = _build_info_icon(aprendiz_helper_text)
                    asg_disp = int(round(float(cargo_sugestao.get("asg", asg_sugerido))))
                    cargo_sugestao_html = (
                        "<div style='margin-top:8px;text-align:center;color:#0c0863;background-color: #f0f2f6; "
                        "border-radius: 10px; padding: 10px 12px; display:flex; "
                        "flex-direction:column; gap:10px;'>"
                        "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;margin-bottom:6px;'>"
                        "Time Comercial</div>"
                        "<div style='display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'>"
                        "<div style='flex:1 1 120px;min-width:120px;'>"
                        f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                        f"Lideres{faixa_helper_icon}</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                        f"{int(round(cargo_sugestao['lideres']))}</div>"
                        f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                        f"({cargo_sugestao['lideres']:.2f} colab.)</div>"
                        "</div>"
                        "<div style='flex:1 1 120px;min-width:120px;'>"
                        f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                        f"Auxiliares{faixa_helper_icon}</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                        f"{int(round(cargo_sugestao['auxiliares']))}</div>"
                        f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                        f"({cargo_sugestao['auxiliares']:.2f} colab.)</div>"
                        "</div>"
                        "<div style='flex:1 1 120px;min-width:120px;'>"
                        f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                        f"Aprend{aprendiz_helper_icon}</div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                        f"{int(round(cargo_sugestao['aprend']))}</div>"
                        f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                        f"({cargo_sugestao['aprend']:.2f} colab.)</div>"
                        "</div>"
                        "<div style='flex:1 1 120px;min-width:120px;'>"
                        "<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                        "ASG <span title='Pedidos/Dia > 162.94 ou Area Total >= 250.00' "
                        "style='margin-left:4px;display:inline-flex;vertical-align:middle;color:#6c6c6c;'>"
                        "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
                        "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
                        "stroke-linejoin='round' class='icon'>"
                        "<circle cx='12' cy='12' r='10'></circle>"
                        "<path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'></path>"
                        "<line x1='12' y1='17' x2='12.01' y2='17'></line>"
                        "</svg></span></div>"
                        f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                        f"{asg_disp}</div>"
                        f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                        f"({float(asg_disp):.2f} colab.)</div>"
                        "</div>"
                        "</div>"
                    )
                seg_sugestao_html = (
                    "<div style='margin-top:8px;text-align:center;color:#0c0863;background-color: #eef6ff; "
                    "border-radius: 10px; padding: 10px 12px; display:flex; "
                    "flex-direction:column; gap:10px;'>"
                    "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;margin-bottom:6px;'>"
                    "Fora do Time Comercial</div>"
                    "<div style='display:flex;flex-wrap:wrap;gap:12px;justify-content:center;align-items:center;'>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    f"Estoquistas{estoq_helper_icon}</div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>{estoq_pred_int}</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                    f"({estoq_pred_raw:.2f} colab.)</div>"
                    "</div>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    "<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    "Seguranca <span title='Pedidos/Dia > 162.94 ou Area Total >= 250.00' "
                    "style='margin-left:4px;display:inline-flex;vertical-align:middle;color:#6c6c6c;'>"
                    "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
                    "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
                    "stroke-linejoin='round' class='icon'>"
                    "<circle cx='12' cy='12' r='10'></circle>"
                    "<path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'></path>"
                    "<line x1='12' y1='17' x2='12.01' y2='17'></line>"
                    "</svg></span></div>"
                    "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                    "&#128274;</div>"
                    "</div>"
                    "</div>"
                    "</div>"
                )
            col_res = st.columns(3)
            with col_res[0]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Time Comercial Historico</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:600;\">{pred_hist_int} colaboradores</div>"
                    f"<div style=\"font-size:0.95rem;font-weight:400;color:#6c6c6c;\">{pred_hist_raw:.2f} colab.</div>"
                    f"<div style=\"font-size:0.9rem;font-weight:400;color:#6c6c6c;\">{criterio_label}: {receita_aux_hist_disp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if ci_hist:
                    low_txt = _format_interval_value(ci_hist.get("ci_low_disp", ci_hist.get("ci_low")))
                    high_txt = _format_interval_value(ci_hist.get("ci_high_disp", ci_hist.get("ci_high")))
                    st.markdown(
                    f"<div style='text-align:center;font-size:0.85rem;color:#555;'>IC 95%: {low_txt} - {high_txt}</div>",
                    unsafe_allow_html=True,
                )
            with col_res[1]:
                st.markdown(
                    f"<div style='text-align:center;color:#0c0863;background-color: #f0f2f6; border-radius: 10px; padding-bottom: 10px;'>"
                    f"<div style='font-size:1.3rem;font-weight:600;'>Time Comercial Ideal</div>"
                    f"<div style='font-size:clamp(1.4rem, 3vw, 2.0rem);font-weight:700; line-height: 0.85;'>"
                    f"{pred_ideal_int} colaboradores</div>"
                    f"<div style='font-size:1.0rem;font-weight:400;color:#6c6c6c;'>({pred_ideal_raw:.2f} colab.)</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>{criterio_label}: {receita_aux_ideal_disp}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if ci_ideal:
                    low_txt = _format_interval_value(ci_ideal.get("ci_low_disp", ci_ideal.get("ci_low")))
                    high_txt = _format_interval_value(ci_ideal.get("ci_high_disp", ci_ideal.get("ci_high")))
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.85rem;color:#555;'>IC 95%: {low_txt} - {high_txt}</div>",
                        unsafe_allow_html=True,
                    )
            with col_res[2]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Diferença (ideal - hist)</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:600;\">{int(round(diff_val)):+d} colaboradores</div>"
                    f"<div style=\"font-size:0.95rem;font-weight:400;color:#6c6c6c;\">{diff_val:+.2f} colab.</div>"
                    f"{urgency_badge}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if cargo_sugestao_html or seg_sugestao_html:
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:12px;margin:14px 0 6px 0;'>"
                    "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                    "<div style='font-size:1.05rem;color:#6c6c6c;font-weight:600;'>Sugestões por Cargo</div>"
                    "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                col_sug_1, col_sug_2 = st.columns([2, 1])
                with col_sug_1:
                    if cargo_sugestao_html:
                        st.markdown(cargo_sugestao_html, unsafe_allow_html=True)
                with col_sug_2:
                    if seg_sugestao_html:
                        st.markdown(seg_sugestao_html, unsafe_allow_html=True)
            similar_df = _select_similar_lojas(
                train_df,
                features_input_ideal_ml or features_input_ideal,
                loja_nome_alvo_submit,
                target_aux_hist=pred_hist_raw,
                n=3,
            )
            if not similar_df.empty:
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:12px;margin:14px 0 6px 0;'>"
                    "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                    "<div style='font-size:1.05rem;color:#6c6c6c;font-weight:600;'>"
                    "Lojas parecidas com a loja calculada</div>"
                    "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                def _predict_ideal_for_row(row_dict: Dict[str, object]) -> Optional[float]:
                    if model_bundle_ideal is None:
                        return None
                    resultados, _ = gerar_resultados_modelos(
                        model_bundle_ideal,
                        train_df,
                        row_dict,
                        "ideal",
                        horas_disp,
                        margem,
                        algo_order=["catboost"],
                        anchor_quantile=anchor_quantile,
                        apply_cluster_blend=False,
                        compute_metrics=False,
                        skip_cap_cols=skip_cap_cols_ideal,
                    )
                    if not resultados:
                        return None
                    pred_val = None
                    for res in resultados:
                        if res.get("key") == "catboost":
                            pred_val = res.get("pred")
                            break
                    if pred_val is None:
                        pred_val = resultados[0].get("pred")
                    try:
                        pred_float = float(pred_val)
                    except Exception:
                        return None
                    return pred_float if math.isfinite(pred_float) else None

                sim_cols = st.columns(3)
                for idx, (_, row) in enumerate(similar_df.iterrows()):
                    if idx >= len(sim_cols):
                        break
                    with sim_cols[idx]:
                        loja_label = str(row.get("Loja", f"Loja {idx + 1}")).strip() or f"Loja {idx + 1}"
                        qtd_aux_hist = safe_float(row.get("QtdAux"), float("nan"))
                        qtd_aux_hist_int = int(round(qtd_aux_hist)) if math.isfinite(qtd_aux_hist) else None
                        ideal_pred = _predict_ideal_for_row(row.to_dict())
                        ideal_pred_int = int(round(ideal_pred)) if ideal_pred is not None else None
                        diff_calc = ideal_pred - qtd_aux_hist if ideal_pred is not None and math.isfinite(qtd_aux_hist) else None
                        diff_disp = f"{diff_calc:+.2f}" if diff_calc is not None and math.isfinite(diff_calc) else "-"
                        urgency_label = _delta_urgency_label(diff_calc)
                        urgency_color = _delta_urgency_color(urgency_label)
                        receita_total_loja = safe_float(row.get("ReceitaTotalMes"), float("nan"))
                        criterio_source_key = "SalarioMapeado" if criterio_is_iaf else criterio_key
                        criterio_denom_sim = safe_float(row.get(criterio_source_key), float("nan"))
                        if not math.isfinite(criterio_denom_sim) or criterio_denom_sim <= 0:
                            criterio_denom_sim = qtd_aux_hist
                        receita_por_aux = None
                        if math.isfinite(receita_total_loja) and math.isfinite(criterio_denom_sim) and criterio_denom_sim > 0:
                            receita_por_aux = receita_total_loja / criterio_denom_sim
                        if criterio_is_iaf:
                            iaf_val = safe_float(row.get("%IAF25"), float("nan"))
                            if math.isfinite(iaf_val):
                                if iaf_val > 1.5:
                                    iaf_val = iaf_val / 100.0
                                if receita_por_aux is not None and math.isfinite(receita_por_aux):
                                    receita_por_aux = receita_por_aux * iaf_val
                        if criterio_key in ("SalarioMapeado", "SalarioMapeadoIAF25"):
                            receita_aux_disp = (
                                f"{receita_por_aux:,.2f}"
                                if receita_por_aux is not None and math.isfinite(receita_por_aux)
                                else "-"
                            )
                        else:
                            receita_aux_disp = (
                                f"R$ {receita_por_aux:,.2f}"
                                if receita_por_aux is not None and math.isfinite(receita_por_aux)
                                else "-"
                            )
                        hist_disp = f"{qtd_aux_hist_int}" if qtd_aux_hist_int is not None else "-"
                        ideal_disp = f"{ideal_pred_int}" if ideal_pred_int is not None else "-"
                        st.markdown(
                            "<div style='border:1px solid #e5e7f0;border-radius:12px;padding:12px;background:#fff;'>"
                            "<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:6px;color:#0c0c1f;'>"
                            f"<span style='width:10px;height:10px;background:{urgency_color};border-radius:3px;display:inline-block;'></span>"
                            f"<div style='font-weight:600;'>{loja_label}</div>"
                            "</div>"
                            f"<div style='font-size:0.9rem;color:#555;'>Time Comercial Historico: <b>{hist_disp}</b></div>"
                            f"<div style='font-size:0.9rem;color:#555;'>Time Comercial Ideal: <b>{ideal_disp}</b></div>"
                            f"<div style='font-size:0.9rem;color:#555;'>Diferenca: <b>{diff_disp}</b></div>"
                            f"<div style='font-size:0.85rem;color:#555;'>Urgencia: <b style='color:{urgency_color};'>{urgency_label or '-'}</b></div>"
                            f"<div style='font-size:0.9rem;color:#555;'>{criterio_label} Real: <b>{receita_aux_disp}</b></div>"
                            "</div>"
                            ,
                            unsafe_allow_html=True,
                        )
            if mostrar_metricas:
                metrics_info_ideal = cat_ideal.get("metrics") or {}

                def _render_metric_card(config: Dict[str, object], raw_val: float) -> str:
                    if not _metric_has_value(raw_val):
                        return ""
                    val = config.get("transform", lambda x: x)(float(raw_val))
                    bands = config.get("bands", [])
                    if not bands:
                        return ""
                    formatter = config.get("format", lambda x: f"{x}")
                    scale_max = float(config.get("scale_max", max(b.get("max", 0.0) for b in bands)))
                    if math.isfinite(val):
                        scale_max = max(scale_max, val)
                    scale_max = max(scale_max, 1e-6)
                    gradient_parts: List[str] = []
                    for idx_band, band in enumerate(bands):
                        band_min = float(band.get("min", 0.0))
                        band_max = float(band.get("max", scale_max))
                        if idx_band == len(bands) - 1 and band_max < scale_max:
                            band_max = scale_max
                        start_pct = (band_min / scale_max) * 100.0
                        end_pct = (band_max / scale_max) * 100.0
                        gradient_parts.append(f"{band['color']} {start_pct:.1f}%, {band['color']} {end_pct:.1f}%")
                    grad_css = ", ".join(gradient_parts) if gradient_parts else "#e8edf7 0%, #e8edf7 100%"
                    value_pct = max(0.0, min((val / scale_max) * 100.0 if math.isfinite(val) else 0.0, 100.0))
                    current_band = next(
                        (
                            band for band in bands
                            if val >= float(band.get("min", 0.0)) and val < float(band.get("max", scale_max))
                        ),
                        bands[-1],
                    )
                    legend_bits = []
                    for idx, band in enumerate(bands):
                        band_min = float(band.get("min", 0.0))
                        band_max = float(band.get("max", scale_max))
                        if idx == len(bands) - 1 and band_max < scale_max:
                            band_max = scale_max
                        if idx == len(bands) - 1 and band_max >= scale_max:
                            range_txt = f">= {formatter(band_min)}"
                        else:
                            range_txt = f"{formatter(band_min)} - {formatter(band_max)}"
                        legend_bits.append(f"{band['label']}: {range_txt}")
                    faixa_otima = str(config.get("faixa_otima") or "")
                    helper = str(config.get("helper") or "")
                    current_color = current_band.get("color", "#0c0863")
                    badge = (
                        f"<span style='background:{current_color}22;color:{current_color};"
                        f"padding:2px 8px;border-radius:12px;font-weight:700;'>"
                        f"{current_band.get('label', '')}</span>"
                    )
                    return f"""
<div style="border:1px solid #e5e7f0;border-radius:12px;padding:12px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.03);margin-bottom:8px;">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem;flex-wrap:wrap;">
    <div style="font-weight:600;color:#0c0c1f;">{config.get('label')}</div>
    <div style="font-size:1.1rem;font-weight:700;color:#0c0863;">{formatter(val)}</div>
  </div>
  <div style="font-size:0.85rem;color:#505565;margin:4px 0 8px 0;">Faixa ótima: {faixa_otima}</div>
  <div style="position:relative;height:12px;border-radius:999px;overflow:hidden;background:#eef1f7;">
    <div style="position:absolute;inset:0;background:linear-gradient(90deg,{grad_css});"></div>
    <div style="position:absolute;left:{value_pct:.1f}%;top:-1px;transform:translateX(-50%);">
      <div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid #0c0863;"></div>
    </div>
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;font-size:0.82rem;color:#505565;gap:8px;flex-wrap:wrap;">
    <div style="flex:1 1 60%;line-height:1.3;">{" | ".join(legend_bits)}</div>
    <div>{badge}</div>
  </div>
  <div style="font-size:0.82rem;color:#6c6c6c;margin-top:6px;line-height:1.35;">{helper}</div>
</div>
"""

                metric_configs: List[Dict[str, object]] = [
                    {
                        "key": "SMAPE",
                        "label": "Precisao",
                        "transform": lambda v: (1.0 - v) * 100.0,
                        "format": lambda v: f"{v:.1f}%",
                        "bands": [
                            {"label": "Crítico", "min": 0.0, "max": 50.0, "color": "#d8516d"},
                            {"label": "Ajustar", "min": 50.0, "max": 70.0, "color": "#f0b429"},
                            {"label": "Bom", "min": 70.0, "max": 85.0, "color": "#4da3f5"},
                            {"label": "Ótimo", "min": 85.0, "max": 100.0, "color": "#2c9a6c"},
                        ],
                        "faixa_otima": "Ótimo >= 85% de acerto (1 - SMAPE)",
                        "helper": "Percentual de previsoes com erro baixo (1 - SMAPE). Quanto maior, melhor para confianca no headcount.",
                        "scale_max": 100.0,
                    },
                    {
                        "key": "MAE",
                        "label": "MAE",
                        "format": lambda v: f"{v:.2f}",
                        "bands": [
                            {"label": "Ótimo", "min": 0.0, "max": 1.0, "color": "#2c9a6c"},
                            {"label": "Bom", "min": 1.0, "max": 2.0, "color": "#4da3f5"},
                            {"label": "Atencao", "min": 2.0, "max": 3.0, "color": "#f0b429"},
                            {"label": "Alto", "min": 3.0, "max": 5.0, "color": "#d8516d"},
                        ],
                        "faixa_otima": "Ótimo < 1 auxiliar de erro medio",
                        "helper": "Erro absoluto medio em colaboradores. Indica a diferenca media entre previsto e observado. Quanto menor, melhor.",
                        "scale_max": 5.0,
                    },
                    {
                        "key": "R2_mean",
                        "label": "R2",
                        "format": lambda v: f"{v:.2f}",
                        "bands": [
                            {"label": "Fraco", "min": 0.0, "max": 0.50, "color": "#d8516d"},
                            {"label": "Ok", "min": 0.50, "max": 0.70, "color": "#f0b429"},
                            {"label": "Bom", "min": 0.70, "max": 0.85, "color": "#4da3f5"},
                            {"label": "Ótimo", "min": 0.85, "max": 1.0, "color": "#2c9a6c"},
                        ],
                        "faixa_otima": "Ótimo >= 0.85 explicando variacao",
                        "helper": "Proporcao da variacao explicada pelo modelo (1.0 é perfeito). Quanto maior, melhor a aderencia aos dados reais.",
                        "scale_max": 1.0,
                    },
                    {
                        "key": "RMSE",
                        "label": "RMSE",
                        "format": lambda v: f"{v:.2f}",
                        "bands": [
                            {"label": "Ótimo", "min": 0.0, "max": 1.5, "color": "#2c9a6c"},
                            {"label": "Bom", "min": 1.5, "max": 2.5, "color": "#4da3f5"},
                            {"label": "Atencao", "min": 2.5, "max": 3.5, "color": "#f0b429"},
                            {"label": "Alto", "min": 3.5, "max": 5.5, "color": "#d8516d"},
                        ],
                        "faixa_otima": "Ótimo ate 1.5 colaboradores",
                        "helper": "Raiz do erro quadratico medio. Penaliza mais os erros grandes e indica estabilidade do modelo. Quanto menor, melhor.",
                        "scale_max": 5.5,
                    },
                ]

                metric_cards: List[str] = []
                for cfg in metric_configs:
                    raw_val = metrics_info_ideal.get(cfg["key"])
                    card_html = _render_metric_card(cfg, raw_val)
                    if card_html:
                        metric_cards.append(card_html)

                if metric_cards:
                    st.markdown(
                        "<div style='display:flex;align-items:center;gap:12px;margin:14px 0 6px 0;'>"
                        "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                        "<div style='font-size:1.05rem;color:#6c6c6c;font-weight:600;'>"
                        "Qualidade do modelo</div>"
                        "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Faixas referenciais para previsao de headcount; a seta mostra onde o modelo atual está em cada indicador.")
                    for idx in range(0, len(metric_cards), 2):
                        cols = st.columns(2)
                        for col, card_html in zip(cols, metric_cards[idx : idx + 2]):
                            with col:
                                st.markdown(card_html, unsafe_allow_html=True)
                    warn_list = metrics_info_ideal.get("warnings")
                    if warn_list:
                        st.caption("Avisos do modelo: " + " | ".join(map(str, warn_list)))
        else:
            st.info("Modelo CatBoost indisponivel para historico ou ideal.")

        if model_errors_hist:
            itens = []
            for key, msg in (model_errors_hist or {}).items():
                if key == "_geral":
                    itens.append(msg)
                else:
                    itens.append(f"{MODEL_ALGO_NAMES.get(key, key)}: {msg}")
            st.info("Modelos indisponiveis ou com erro (Historico): " + "; ".join(itens))
        if model_errors_ideal:
            itens = []
            for key, msg in (model_errors_ideal or {}).items():
                if key == "_geral":
                    itens.append(msg)
                else:
                    itens.append(f"{MODEL_ALGO_NAMES.get(key, key)}: {msg}")
            st.info("Modelos indisponiveis ou com erro (Ideal): " + "; ".join(itens))
    elif modo_simplificado:
        sim_inputs = st.session_state.get("sim_inputs", {})
        processos_freq_dict = st.session_state.get("sim_processos_freq", {}) or {}
        tempo_loja_dict = st.session_state.get("sim_processos_tempos_loja", {}) or {}
        tempo_global_dict = st.session_state.get("sim_processos_tempos_global", {}) or {}
        tempo_custom_dict = st.session_state.get("sim_processos_tempos_custom", {}) or {}
        if tempo_custom_dict:
            tempo_loja_dict = dict(tempo_loja_dict)
            tempo_global_dict = dict(tempo_global_dict)
            for proc_norm, tempo_val in tempo_custom_dict.items():
                try:
                    tempo_float = float(tempo_val)
                except Exception:
                    continue
                tempo_float = max(0.0, tempo_float)
                tempo_loja_dict[proc_norm] = tempo_float
                tempo_global_dict[proc_norm] = tempo_float
        estrutura_flags = {"Escritorio": int(escritorio), "Copa": int(copa), "Espaco Evento": int(espaco_evento)}
        result_ideal = calcular_resultado_ideal_simplificado(
            cluster_values=cluster_values,
            sim_inputs=sim_inputs,
            horas_loja=horas_loja,
            horas_por_colab=horas_por_colab,
            dias_operacionais_ativos=dias_operacionais_ativos,
            fator_monotonia=fator_monotonia,
            margem=margem,
            sla_buffer=sla_buffer,
            ocupacao_alvo=ocupacao_alvo,
            absenteismo=absenteismo,
            area_total=area_total,
            qtd_caixas=qtd_caixas,
            estrutura_flags=estrutura_flags,
            base_ativa=base_ativa,
            receita_total=receita_total,
            pct_retirada_hist=pct_retirada,
            itens_pedido_hist=itens_pedido,
            faturamento_hora_hist=cluster_values.get("Faturamento/Hora", 0.0),
            processos_freq_dict=processos_freq_dict,
            tempo_loja_dict=tempo_loja_dict,
            tempo_global_dict=tempo_global_dict,
        )
    else:
        st.error(f"Modo de calculo nao reconhecido: {modo_calc}")
        result_ideal = None

    if modo_simplificado and result_ideal is not None:
        st.success("Calculo (Ideal) concluido!")

        qtd_aux_atual = None
        lookup_row = st.session_state.get("lookup_row")
        if st.session_state.get("lookup_found") and isinstance(lookup_row, dict):
            qtd_aux_atual = safe_float(get_lookup_value("TotalMapeado", ["QtdAux"]))

        ideal_val = float(result_ideal.get("qtd_aux_ideal", 0.0))
        cargo_sugestao_html = ""
        seg_sugestao_html = ""
        if mostrar_sugestoes_cargo:
            estoq_feature_row: Dict[str, object] = {}
            lookup_row = st.session_state.get("lookup_row") or {}
            if isinstance(lookup_row, dict):
                estoq_feature_row.update(lookup_row)
            estoq_feature_row.update(
                {
                    "BaseAtiva": base_ativa,
                    "Pedidos/Dia": pedidos_dia,
                    "Pedidos/Hora": pedidos_hora,
                    "Itens/Pedido": itens_pedido,
                    "Faturamento/Hora": faturamento_hora,
                    "ReceitaTotalMes": receita_total,
                    "%Retirada": pct_retirada,
                }
            )
            estoq_pred_raw, estoq_helper_text = _predict_estoquistas_sugestao(
                estoq_train_df, estoq_feature_row
            )
            estoq_pred_int = int(round(estoq_pred_raw))
            estoq_helper_icon = _build_info_icon(estoq_helper_text)
            asg_sugerido = _suggest_staff_from_pedidos_dia(pedidos_dia, area_total)
            seg_sugerido = _suggest_staff_from_pedidos_dia(pedidos_dia, area_total)
            cargo_sugestao = _compute_cargo_suggestion(
                pessoas_df,
                st.session_state.get("fIndicadores"),
                ideal_val,
                base_ativa,
                pedidos_dia,
                area_total,
                ASG_SEG_PEDIDOS_DIA_THRESHOLD,
                ASG_SEG_AREA_TOTAL_THRESHOLD,
            )
            if cargo_sugestao:
                total_aprendizes = _resolve_total_colabs_for_aprendizes(
                    st.session_state.get("lookup_row"),
                    pessoas_df,
                    st.session_state.get("fIndicadores"),
                    base_ativa,
                    receita_total,
                )
                cargo_sugestao, aprendiz_helper_text = _apply_aprendizes_rule(
                    cargo_sugestao, total_aprendizes
                )
                faixa_helper_text = (
                    f"Faixa BaseAtiva: {cargo_sugestao['faixa_label']} | "
                    f"Aux/Lider: {cargo_sugestao['aux_lider']:.2f}"
                )
                faixa_helper_icon = _build_info_icon(faixa_helper_text)
                aprendiz_helper_icon = _build_info_icon(aprendiz_helper_text)
                asg_disp = int(round(float(cargo_sugestao.get("asg", asg_sugerido))))
                cargo_sugestao_html = (
                    "<div style='margin-top:8px;text-align:center;color:#0c0863;background-color: #f0f2f6; "
                    "border-radius: 10px; padding: 10px 12px; display:flex; "
                    "flex-direction:column; gap:10px;'>"
                    "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;margin-bottom:6px;'>"
                    "Time Comercial</div>"
                    "<div style='display:flex;flex-wrap:wrap;gap:12px;justify-content:center;'>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    f"Lideres{faixa_helper_icon}</div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                    f"{int(round(cargo_sugestao['lideres']))}</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                    f"({cargo_sugestao['lideres']:.2f} colab.)</div>"
                    "</div>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    f"Auxiliares{faixa_helper_icon}</div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                    f"{int(round(cargo_sugestao['auxiliares']))}</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                    f"({cargo_sugestao['auxiliares']:.2f} colab.)</div>"
                    "</div>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    f"Aprend{aprendiz_helper_icon}</div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                    f"{int(round(cargo_sugestao['aprend']))}</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                    f"({cargo_sugestao['aprend']:.2f} colab.)</div>"
                    "</div>"
                    "<div style='flex:1 1 120px;min-width:120px;'>"
                    "<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                    "ASG <span title='Pedidos/Dia > 162.94 ou Area Total >= 250.00' "
                    "style='margin-left:4px;display:inline-flex;vertical-align:middle;color:#6c6c6c;'>"
                    "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
                    "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
                    "stroke-linejoin='round' class='icon'>"
                    "<circle cx='12' cy='12' r='10'></circle>"
                    "<path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'></path>"
                    "<line x1='12' y1='17' x2='12.01' y2='17'></line>"
                    "</svg></span></div>"
                    f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                    f"{asg_disp}</div>"
                    f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                    f"({float(asg_disp):.2f} colab.)</div>"
                    "</div>"
                    "</div>"
                )
            seg_sugestao_html = (
                "<div style='margin-top:8px;text-align:center;color:#0c0863;background-color: #eef6ff; "
                "border-radius: 10px; padding: 10px 12px; display:flex; "
                "flex-direction:column; gap:10px;'>"
                "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;margin-bottom:6px;'>"
                "Fora do Time Comercial</div>"
                "<div style='display:flex;flex-wrap:wrap;gap:12px;justify-content:center;align-items:center;'>"
                "<div style='flex:1 1 120px;min-width:120px;'>"
                f"<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                f"Estoquistas{estoq_helper_icon}</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>{estoq_pred_int}</div>"
                f"<div style='font-size:0.9rem;font-weight:400;color:#6c6c6c;'>"
                f"({estoq_pred_raw:.2f} colab.)</div>"
                "</div>"
                "<div style='flex:1 1 120px;min-width:120px;'>"
                "<div style='font-size:0.95rem;font-weight:600;color:#0c0863;'>"
                "Seguranca <span title='Pedidos/Dia > 162.94 ou Area Total >= 250.00' "
                "style='margin-left:4px;display:inline-flex;vertical-align:middle;color:#6c6c6c;'>"
                "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' "
                "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
                "stroke-linejoin='round' class='icon'>"
                "<circle cx='12' cy='12' r='10'></circle>"
                "<path d='M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3'></path>"
                "<line x1='12' y1='17' x2='12.01' y2='17'></line>"
                "</svg></span></div>"
                "<div style='font-size:1.3rem;font-weight:700;color:#0c0863;'>"
                "&#128274;</div>"
                "</div>"
                "</div>"
                "</div>"
            )
        if qtd_aux_atual is not None and not math.isnan(qtd_aux_atual):
            diff_val = ideal_val - float(qtd_aux_atual)
            urgency_badge = _render_urgency_badge(diff_val)
            col_res = st.columns(3)
            with col_res[0]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                f"<div style=\"font-size:1.1rem;font-weight:500;\">Time Comercial Atual</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:500;\">{qtd_aux_atual:.2f} colab.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_res[1]:
                st.markdown(
                    f"<div style='text-align:center;color:#0c0863;background-color: #f0f2f6; border-radius: 10px; padding-bottom: 10px;'>"
                f"<div style='font-size:1.3rem;font-weight:600;'>Time Comercial Ideal</div>"
                    f"<div style='font-size:clamp(1.4rem, 3vw, 2.0rem);font-weight:600; line-height: 0.85;'>"
                    f"{ideal_val:.2f} colab.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_res[2]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Diferenca (ideal - atual)</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:500;\">{diff_val:+.2f} colab.</div>"
                    f"{urgency_badge}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.metric("Time Comercial (ideal)", f"{ideal_val}")
        if cargo_sugestao_html or seg_sugestao_html:
            st.markdown(
                "<div style='display:flex;align-items:center;gap:12px;margin:14px 0 6px 0;'>"
                "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                "<div style='font-size:1.05rem;color:#6c6c6c;font-weight:600;'>Sugestões por Cargo</div>"
                "<div style='flex:1;height:1px;background:#e5e7eb;'></div>"
                "</div>",
                unsafe_allow_html=True,
            )
            col_sug_1, col_sug_2 = st.columns([2, 1])
            with col_sug_1:
                if cargo_sugestao_html:
                    st.markdown(cargo_sugestao_html, unsafe_allow_html=True)
            with col_sug_2:
                if seg_sugestao_html:
                    st.markdown(seg_sugestao_html, unsafe_allow_html=True)

        st.caption(
            f"Carga: {result_ideal['carga_total_horas']:.2f} h/semana | "
            f"H/colab efetivo: {result_ideal['horas_por_colaborador']:.2f} h/semana "
            f"(base {result_ideal.get('horas_por_colaborador_base', result_ideal['horas_por_colaborador']):.2f}) | "
            f"Ocupacao alvo: {result_ideal['ocupacao_alvo']:.2f} | "
            f"Absenteismo: {result_ideal['absenteismo']:.2f} | "
            f"SLA buffer: {result_ideal['sla_buffer']:.2f} | "
            f"Margem: {result_ideal['margem']:.2f}"
        )
        st.caption(
            f"Carga (fluxo): {result_ideal.get('carga_fluxo', 0.0):.2f} h/sem | "
            f"Carga (processos extras): {result_ideal.get('carga_processos_extras', 0.0):.2f} h/sem"
        )
        st.caption(
            f"Pedidos/h usados: {result_ideal.get('pedidos_hora_utilizado', 0.0):.2f} | "
            f"Tempo medio: {result_ideal.get('tmedio_min_atendimento', 0.0):.2f} min | "
            f"Fator monotonia: {result_ideal.get('fator_monotonia', fator_monotonia):.2f}"
        )
# =============================================================================
# Helpers internos (fila)
# =============================================================================
from src.logic.models.model_fila import (
    estimate_queue_inputs,
    calcular_fila,
    QUEUE_CALIBRATION_DEFAULT,
)


# =============================================================================
# Constantes
# =============================================================================
PROCESSOS_PRIORITARIOS = [
    "Devolução",
    "Reposição de prateleira",
    "Produção de flyer",
    "Abertura e acompanhamento de chamado",
    "Ação de VPs/Excesso",
    "Criação de conteúdo",
    "Elaboração de calendário do ciclo e divulgação",
    "Encontro de ciclo",
    "Eventos para os revendedores",
    "Unibê",
    "Limpeza da ER",
    "Limpeza das salas e Copa",
    "Limpeza dos banheiros",
    "Mudança de planograma",
    "Fechamento de caixa",
    "Atualização de cadastro de revendedor",
    "Cadastro de revendedor",
    "Digitalização de boletos",
    "Faturamente de pedido (retirada e delivery)",
    "Separação de mercadoria (on-line e retirada)",
    "Venda em caixa",
    "Atendimento ao cliente",
    "Prospecção de revendedor (Início)",
    "Atendimento online",
    "Ativações on-line",
]


def _render_queue_comparison_block(
    resultados_modelos: List[Dict[str, object]],
    feature_row: Optional[Dict[str, object]],
    rho_target: float,
    contexto: str = "",
) -> None:
    # Não exibir bloco de fila no modo Ideal (pedido do usuário)
    if contexto and "Ideal" in contexto:
        return
    if not resultados_modelos or not feature_row:
        return
    queue_inputs = estimate_queue_inputs(feature_row)
    lambda_h = queue_inputs.get("lambda_hora", 0.0)
    tma_min = queue_inputs.get("tma_min", 0.0)
    mu_h = queue_inputs.get("mu_hora", 0.0)
    rho_target_use = rho_target if _metric_has_value(rho_target) else DEFAULT_OCUPACAO_ALVO
    calibration_factor = QUEUE_CALIBRATION_DEFAULT
    fila_diag = calcular_fila(
        lambda_h,
        tma_min,
        rho_target=rho_target_use,
        calibration_factor=calibration_factor,
    )
    c_fila = float(fila_diag.get("c_fila", 0.0) or 0.0)
    rho_fila = fila_diag.get("rho_fila")
    rho_target_disp = fila_diag.get("rho_target", rho_target_use)
    st.subheader(f"Modelo Teoria das Filas{contexto}")
    fila_text = f"Headcount (fila): {c_fila:.2f} colaboradores"
    if _metric_has_value(rho_fila):
        fila_text += f" (ρ ≈ {rho_fila:.2f}, alvo ρ_target={rho_target_disp:.2f})"
    st.write(fila_text)

    st.subheader(f"Comparação vs fila{contexto}")
    for res in resultados_modelos:
        label = res.get("label", res.get("key", "Modelo"))
        c_raw = float(res.get("pred", 0.0))
        c_pred = int(round(c_raw))
        rho_val = float("nan")
        if mu_h > 0 and c_pred > 0:
            rho_val = lambda_h / (c_pred * mu_h)
        delta_abs = float(c_pred - c_fila)
        line = f"{label}: {c_pred} colaboradores"
        extras: List[str] = []
        if _metric_has_value(rho_val):
            extras.append(f"ρ ≈ {rho_val:.2f}")
        if c_fila > 0:
            delta_pct = delta_abs / c_fila
            extras.append(f"Δ vs fila: {delta_abs:+.2f} colab. ({delta_pct*100:+.1f}%)")
        elif delta_abs != 0:
            extras.append(f"Δ vs fila: {delta_abs:+.2f} colab.")
        if extras:
            line += f" ({'; '.join(extras)})"
        st.write(line)

    rho_hist = float("nan")
    diag_bits: List[str] = []
    if _metric_has_value(lambda_h):
        diag_bits.append(f"λ≈{lambda_h:.2f}/h")
    if _metric_has_value(tma_min):
        diag_bits.append(f"TMA≈{tma_min:.1f} min")
    if _metric_has_value(mu_h):
        diag_bits.append(f"μ≈{mu_h:.2f}/h")
    if math.isfinite(rho_hist):
        diag_bits.append(f"ρ_hist≈{rho_hist:.2f}")
    diag_bits.append(f"fator≈{calibration_factor:.2f}")
    if diag_bits:
        st.caption(" | ".join(diag_bits))
    rho_debug = rho_fila if _metric_has_value(rho_fila) else float("nan")
    st.caption(
        "[DEBUG FILA] "
        f"λ={lambda_h:.2f}/h, TMA={tma_min:.2f} min, μ={mu_h:.2f}/h, "
        f"fator={calibration_factor:.2f}, c_fila={c_fila:.2f}, ρ={rho_debug:.2f}"
    )
    cat_ci = next(
        (res.get("ci_debug") for res in resultados_modelos if res.get("key") == "catboost" and res.get("ci_debug")),
        None,
    )
    if cat_ci:
        st.caption(
            f"CatBoost (p5/p50/p95 pré-fila): "
            f"{cat_ci['ci_low']:.2f} / {cat_ci['pred_mean']:.2f} / {cat_ci['ci_high']:.2f}"
        )
