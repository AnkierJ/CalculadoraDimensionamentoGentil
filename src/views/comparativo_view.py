import pandas as pd
import streamlit as st
from typing import Dict, Optional, List

from src.logic.core.logic import (
    prepare_training_dataframe,
    clean_training_dataframe,
    gerar_resultados_modelos,
    _train_cached,
    _compute_porte_cluster_context,
    _assign_porte_cluster,
    _is_loja_grande,
)
from src.logic.data.buscaDeLojas import _get_loja_row, _ensure_loja_key
from src.logic.utils.helpers import safe_float


def _preds_for_loja(
    bundle: Optional[Dict[str, object]],
    train_df: pd.DataFrame,
    feature_row: Dict[str, object],
    ref_mode: str,
    horas_disp: float,
    margem: float,
    anchor_quantile: float,
) -> Dict[str, float]:
    """Retorna previsões por algoritmo para a loja informada."""
    if bundle is None or not feature_row:
        return {}
    resultados, _ = gerar_resultados_modelos(
        bundle,
        train_df,
        feature_row,
        ref_mode,
        horas_disp,
        margem,
        anchor_quantile=anchor_quantile,
        apply_cluster_blend=False,
    )
    preds: Dict[str, float] = {}
    for res in resultados:
        if res.get("pred") is None:
            continue
        try:
            preds[res.get("key", "")] = float(res.get("pred"))
        except Exception:
            continue
    return preds


def render_comparativo_tab(tab_container) -> None:
    with tab_container:
        st.subheader("Comparativo Histórico vs Ideal (lojas)")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            n_lojas = st.number_input(
                "Número de lojas",
                min_value=1,
                max_value=40,
                value=int(st.session_state.get("comparativo_n_lojas", 20)),
                step=1,
                help="Quantidade de lojas do topo da base de estrutura para comparar.",
            )
            st.session_state["comparativo_n_lojas"] = int(n_lojas)
        with col2:
            anchor_percent = st.select_slider(
                "Âncora receita/aux (%)",
                options=[50, 55, 60, 65, 70, 75, 80, 85, 90],
                value=int(st.session_state.get("anchor_rpa_percent", 60)),
                help="Percentil de receita por auxiliar usado como referência: se a meta é evitar falta de gente, prefira percentil mais baixo; se a meta é eficiência agressiva, percentil mais alto.",
            )
            st.session_state["anchor_rpa_percent"] = anchor_percent
            anchor_quantile = float(anchor_percent) / 100.0
            st.session_state["anchor_rpa_quantile"] = anchor_quantile
        with col3:
            st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
            submitted = st.button("Realizar Comparativo", type="primary", use_container_width=True)

        estrutura_df = st.session_state.get("dEstrutura")
        pessoas_df = st.session_state.get("dPessoas")
        indicadores_df = st.session_state.get("fIndicadores")
        if estrutura_df is None or estrutura_df.empty:
            st.warning("Base de estrutura vazia ou não carregada.")
            return

        if not submitted:
            st.info("Informe os parâmetros e clique em Realizar Comparativo.")
            return

        train_df = prepare_training_dataframe(estrutura_df, pessoas_df, indicadores_df)
        train_df = clean_training_dataframe(train_df)
        if train_df.empty:
            st.warning("Sem dados válidos para treinar os modelos (verifique dEstrutura/dPessoas/fIndicadores).")
            return

        estrutura_norm = _ensure_loja_key(estrutura_df)
        train_norm = _ensure_loja_key(train_df)
        if "Loja" not in estrutura_norm.columns:
            st.warning("Coluna 'Loja' não encontrada na base de estrutura.")
            return

        horas_disp = float(st.session_state.get("horas_disp_semanais", 44.0))
        margem = float(st.session_state.get("folga_operacional", 0.15))

        cluster_ctx = _compute_porte_cluster_context(
            train_df,
            mode="historico",
            horas_disp=horas_disp,
            margem=margem,
            anchor_quantile=anchor_quantile,
        )

        def _classificar_porte(row_dict: Dict[str, object]) -> tuple[str, Optional[Dict[str, object]]]:
            if not cluster_ctx:
                return "Loja média", None
            porte_map = cluster_ctx.get("porte_map", {}) or {}
            cid, c_pred, c_size = _assign_porte_cluster(row_dict, cluster_ctx)
            porte_code = porte_map.get(cid)
            thresholds = cluster_ctx.get("thresholds", {}) or {}
            ratios = []
            for col, thr in thresholds.items():
                val = safe_float(row_dict.get(col), 0.0)
                if thr and thr > 0 and val >= 0:
                    ratios.append(val / thr)
            max_ratio = max(ratios) if ratios else 0.0
            # Se porte_code existir, seguir mapeamento: 1=grande, 2/3=média, 4=pequena
            if porte_code is not None:
                if porte_code == 1:
                    porte = "Loja grande"
                elif porte_code in (2, 3):
                    porte = "Loja média"
                else:
                    porte = "Loja pequena"
            else:
                is_large = _is_loja_grande(row_dict, thresholds)
                porte = "Loja grande" if is_large or max_ratio >= 1.0 else ("Loja pequena" if max_ratio <= 0.5 else "Loja média")
            cid, c_pred, c_size = _assign_porte_cluster(row_dict, cluster_ctx)
            info = {
                "cluster_id": cid,
                "cluster_pred": c_pred,
                "cluster_size": c_size,
                "max_ratio": max_ratio,
                "porte_code": porte_code,
            }
            return porte, info

        lojas_top: List[str] = estrutura_norm["Loja"].astype(str).head(int(n_lojas)).tolist()

        cache_ver = 9 + int(anchor_quantile * 100)
        model_hist = _train_cached(
            train_df,
            "historico",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver,
        )
        model_ideal = _train_cached(
            train_df,
            "ideal",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver,
        )

        linhas_comp: List[Dict[str, object]] = []
        for loja_nome in lojas_top:
            feature_row, _ = _get_loja_row(train_norm, loja_nome)
            if not feature_row:
                continue
            loja_display = str(feature_row.get("Loja", loja_nome)).strip() or loja_nome
            porte_label, porte_info = _classificar_porte(feature_row)
            preds_hist = _preds_for_loja(model_hist, train_df, feature_row, "historico", horas_disp, margem, anchor_quantile)
            preds_ideal = _preds_for_loja(model_ideal, train_df, feature_row, "ideal", horas_disp, margem, anchor_quantile)
            qtd_hist_cat = preds_hist.get("catboost")
            qtd_hist_xg = preds_hist.get("xgboost")
            qtd_hist_pref = qtd_hist_cat if qtd_hist_cat is not None else qtd_hist_xg
            if qtd_hist_pref is None:
                qtd_hist_pref = safe_float(feature_row.get("QtdAux"))
            qtd_ideal_cat = preds_ideal.get("catboost")
            qtd_ideal_xg = preds_ideal.get("xgboost")
            qtd_ideal_pref = qtd_ideal_cat if qtd_ideal_cat is not None else qtd_ideal_xg
            receita = safe_float(feature_row.get("ReceitaTotalMes"))
            receita_por_aux = None
            if receita and qtd_hist_pref and qtd_hist_pref > 0:
                receita_por_aux = receita / qtd_hist_pref
            delta_ideal = None
            if qtd_ideal_pref is not None and qtd_hist_pref is not None:
                delta_ideal = float(qtd_ideal_pref) - float(qtd_hist_pref)
            delta_cat = None
            if qtd_ideal_cat is not None and qtd_hist_cat is not None:
                delta_cat = float(qtd_ideal_cat) - float(qtd_hist_cat)
            delta_xg = None
            if qtd_ideal_xg is not None and qtd_hist_xg is not None:
                delta_xg = float(qtd_ideal_xg) - float(qtd_hist_xg)
            linhas_comp.append(
                {
                    "Loja": loja_display,
                    "QtdAux Histórico": qtd_hist_pref,
                    "QtdAux Ideal": qtd_ideal_pref,
                    "QtdAux Ideal CatBoost": qtd_ideal_cat,
                    "QtdAux Ideal XGBoost": qtd_ideal_xg,
                    "Delta CatBoost (Ideal-Hist)": delta_cat,
                    "Delta XGBoost (Ideal-Hist)": delta_xg,
                    "QtdIdeal - QtdHistórico": delta_ideal,
                    "Porte": porte_label,
                    "Cluster Porte": porte_info.get("porte_code") if porte_info else None,
                    "Faturamento/Aux (hist)": receita_por_aux,
                }
            )

        if not linhas_comp:
            st.info("Não há linhas suficientes para exibir o comparativo.")
            return

        tabela_comp = pd.DataFrame(linhas_comp)
        for porte in ("Loja grande", "Loja média", "Loja pequena"):
            subset = tabela_comp.loc[tabela_comp["Porte"] == porte].reset_index(drop=True)
            if subset.empty:
                continue
            st.subheader(porte)
            st.dataframe(subset.drop(columns=["Porte"]), use_container_width=True)
