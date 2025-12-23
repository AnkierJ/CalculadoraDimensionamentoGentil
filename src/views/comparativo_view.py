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
        compute_metrics=False,  # evita CV pesado repetido no comparativo
        algo_order=["catboost"],  # usa apenas o modelo exibido
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
        estrutura_df = st.session_state.get("dEstrutura")
        pessoas_df = st.session_state.get("dPessoas")
        indicadores_df = st.session_state.get("fIndicadores")

        st.subheader("Comparativo Histórico vs Ideal (lojas)")
        # Estilo ajustado mais abaixo para resumo das seleções

        select_all_label = "Selecionar todas"
        if "comparativo_lojas_sel" not in st.session_state:
            st.session_state["comparativo_lojas_sel"] = []

        loja_opcoes: List[str] = []
        if estrutura_df is not None and not estrutura_df.empty and "Loja" in estrutura_df.columns:
            loja_opcoes = (
                estrutura_df["Loja"]
                .astype(str)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
        multiselect_options = [select_all_label] + loja_opcoes
        def _on_change_lojas():
            selections = st.session_state.get("comparativo_lojas_sel", [])
            selections = [s for s in selections if s in multiselect_options]
            if select_all_label in selections:
                # se marcou somente "Selecionar todas", marcar todas
                if len(selections) == 1:
                    selections = multiselect_options[:]
                # se desmarcou alguma depois, remove a flag de selecionar todas
                elif len(selections) - 1 < len(loja_opcoes):
                    selections = [s for s in selections if s != select_all_label]
            st.session_state["comparativo_lojas_sel"] = selections
        st.multiselect(
            "Selecionar lojas",
            options=multiselect_options,
            key="comparativo_lojas_sel",
            on_change=_on_change_lojas,
            help="Escolha uma ou mais lojas para o comparativo.",
            placeholder="Clique para selecionar",
        )
        lojas_sel = st.session_state.get("comparativo_lojas_sel", [])

        # Texto resumido para o input seguindo a lógica solicitada
        selected_lojas = [loja for loja in lojas_sel if loja in loja_opcoes]
        has_all_selected = select_all_label in lojas_sel and len(selected_lojas) >= len(loja_opcoes)
        display_lojas = loja_opcoes[:] if has_all_selected else selected_lojas
        display_count = len(display_lojas)

        if has_all_selected:
            summary_text = "Todas as Lojas"
            show_summary_only = True
        elif display_count == 0:
            summary_text = "Selecionar lojas"
            show_summary_only = True
        elif display_count == 1:
            summary_text = display_lojas[0]
            show_summary_only = False
        elif display_count == 2:
            summary_text = ", ".join(display_lojas[:2])
            show_summary_only = False
        else:
            summary_text = "Seleções múltiplas"
            show_summary_only = True
        
        col1, col2, col3 = st.columns([1, 1, 1])
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
        pessoas_norm = _ensure_loja_key(pessoas_df)
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

        lojas_sel = st.session_state.get("comparativo_lojas_sel", [])
        lojas_validas = estrutura_norm["Loja"].astype(str).dropna().drop_duplicates()
        if select_all_label in lojas_sel or not lojas_sel:
            lojas_top: List[str] = lojas_validas.sort_values().tolist()
        else:
            lojas_set = set(lojas_validas.tolist())
            lojas_top = [loja for loja in lojas_sel if loja in lojas_set]
        if not lojas_top:
            st.warning("Selecione ao menos uma loja para continuar.")
            return

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
            def _round_int(val: Optional[float]) -> Optional[int]:
                try:
                    f = float(val)
                    if pd.isna(f):
                        return None
                    return int(round(f))
                except Exception:
                    return None

            qtd_aux_real = None
            if pessoas_norm is not None and not pessoas_norm.empty:
                pessoas_row, _ = _get_loja_row(pessoas_norm, loja_nome)
                if pessoas_row:
                    qtd_aux_real = safe_float(pessoas_row.get("QtdAux"))
            porte_label, porte_info = _classificar_porte(feature_row)
            preds_hist = _preds_for_loja(model_hist, train_df, feature_row, "historico", horas_disp, margem, anchor_quantile)
            preds_ideal = _preds_for_loja(model_ideal, train_df, feature_row, "ideal", horas_disp, margem, anchor_quantile)
            qtd_hist = preds_hist.get("catboost")
            if qtd_hist is None:
                qtd_hist = safe_float(feature_row.get("QtdAux"))
            qtd_ideal = preds_ideal.get("catboost")
            qtd_aux_real_i = _round_int(qtd_aux_real)
            qtd_hist_i = _round_int(qtd_hist)
            qtd_ideal_i = _round_int(qtd_ideal)

            receita = safe_float(feature_row.get("ReceitaTotalMes"))
            receita_por_aux_real = None
            if receita and qtd_aux_real_i and qtd_aux_real_i > 0:
                receita_por_aux_real = receita / qtd_aux_real_i

            delta_ideal = None
            if qtd_ideal_i is not None and qtd_hist_i is not None:
                delta_ideal = int(qtd_ideal_i - qtd_hist_i)
            linhas_comp.append(
                {
                    "Loja": loja_display,
                    "Qtd Aux Real": qtd_aux_real_i,
                    "Qtd Aux Historico": qtd_hist_i,
                    "Qtd Aux Ideal": qtd_ideal_i,
                    "Diferença": delta_ideal,
                    "Faturamento/Qtd Aux Real": receita_por_aux_real,
                    "Porte": porte_label,
                    "Cluster Porte": porte_info.get("porte_code") if porte_info else None,
                }
            )

        if not linhas_comp:
            st.info("Não há linhas suficientes para exibir o comparativo.")
            return

        tabela_comp = pd.DataFrame(linhas_comp)
        def _delta_urgency_label(delta_val: Optional[float]) -> str:
            if delta_val is None or (isinstance(delta_val, float) and pd.isna(delta_val)):
                return ""
            try:
                diff = int(delta_val)
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
                return "#d8516d"
            label_norm = (
                label_text.lower()
                .replace("●", "")
                .strip()
                .replace("ó", "o")
                .replace("ã", "a")
                .replace("ç", "c")
            )
            if "otimo" in label_norm:
                color = "#2c9a6c"
            elif "bom" in label_norm:
                color = "#4da3f5"
            elif "atencao" in label_norm:
                color = "#f0b429"
            else:
                color = "#d8516d"
            return color

        def _format_brl(val: Optional[float]) -> str:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return ""
            try:
                num = float(val)
            except Exception:
                return ""
            formatted = f"{num:,.2f}"
            formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
            return f"R$ {formatted}"

        colunas_saida = [
            "Loja",
            "Qtd Aux Real",
            "Qtd Aux Historico",
            "Qtd Aux Ideal",
            "Diferença",
            "Urgência",
            "Faturamento/Qtd Aux Real",
            "Cluster Porte",
        ]
        for porte in ("Loja grande", "Loja média", "Loja pequena"):
            subset = tabela_comp.loc[tabela_comp["Porte"] == porte].reset_index(drop=True)
            if subset.empty:
                continue
            st.subheader(porte)
            subset_exibir = subset.drop(columns=["Porte"], errors="ignore")
            subset_exibir = subset_exibir[[c for c in colunas_saida if c in subset_exibir.columns]]
            if "Diferença" in subset_exibir.columns:
                subset_exibir = subset_exibir.copy()
                urg_label = subset_exibir["Diferença"].apply(_delta_urgency_label)
                subset_exibir["Urgência"] = urg_label.apply(lambda v: f"● {v}" if v else "")
                styled = subset_exibir.style.format(
                    {"Faturamento/Qtd Aux Real": _format_brl},
                )
                styled = styled.applymap(
                    lambda v: f"color: {_delta_urgency_color(v)}; font-weight: 600;",
                    subset=["Urgência"],
                )
                st.dataframe(styled, use_container_width=True)
            else:
                styled = subset_exibir.style.format(
                    {"Faturamento/Qtd Aux Real": _format_brl},
                )
                st.dataframe(styled, use_container_width=True)
