import pandas as pd
import streamlit as st
from typing import Dict, Optional, List

from src.logic.core.logic import (
    prepare_training_dataframe,
    clean_training_dataframe,
    gerar_resultados_modelos,
    _train_cached,
)
from src.logic.data.buscaDeLojas import _get_loja_row, _ensure_loja_key
from src.logic.utils.helpers import safe_float


def _pred_for_loja(
    bundle: Optional[Dict[str, object]],
    train_df: pd.DataFrame,
    feature_row: Dict[str, object],
    ref_mode: str,
    horas_disp: float,
    margem: float,
    anchor_quantile: float,
) -> Optional[float]:
    if bundle is None or not feature_row:
        return None
    resultados, _ = gerar_resultados_modelos(
        bundle,
        train_df,
        feature_row,
        ref_mode,
        horas_disp,
        margem,
        anchor_quantile=anchor_quantile,
    )
    preferidos = ("catboost", "xgboost")
    for key in preferidos:
        res = next((r for r in resultados if r.get("key") == key and r.get("pred") is not None), None)
        if res:
            return float(res.get("pred"))
    res_any = next((r for r in resultados if r.get("pred") is not None), None)
    if res_any:
        return float(res_any.get("pred"))
    return None


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

        lojas_top: List[str] = estrutura_norm["Loja"].astype(str).head(int(n_lojas)).tolist()
        horas_disp = float(st.session_state.get("horas_disp_semanais", 44.0))
        margem = float(st.session_state.get("folga_operacional", 0.15))

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
            qtd_hist = _pred_for_loja(model_hist, train_df, feature_row, "historico", horas_disp, margem, anchor_quantile)
            qtd_ideal = _pred_for_loja(model_ideal, train_df, feature_row, "ideal", horas_disp, margem, anchor_quantile)
            if qtd_hist is None:
                qtd_hist = safe_float(feature_row.get("QtdAux"))
            delta_ideal = None
            if qtd_ideal is not None and qtd_hist is not None:
                delta_ideal = float(qtd_ideal) - float(qtd_hist)
            linhas_comp.append(
                {
                    "Loja": loja_display,
                    "QtdAux Histórico": qtd_hist,
                    "QtdAux Ideal": qtd_ideal,
                    "QtdIdeal - QtdHistórico": delta_ideal,
                }
            )

        if not linhas_comp:
            st.info("Não há linhas suficientes para exibir o comparativo.")
            return

        tabela_comp = pd.DataFrame(linhas_comp)
        st.dataframe(tabela_comp, use_container_width=True)
