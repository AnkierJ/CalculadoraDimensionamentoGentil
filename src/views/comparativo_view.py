import math
import pandas as pd
import streamlit as st
import unicodedata
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


def _normalize_col_name(name: object) -> str:
    text = "" if name is None else str(name).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(" ", "").replace("_", "").casefold()
    return text


def _find_praca_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if _normalize_col_name(col) == "praca":
            return col
    return None


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
        praca_prefix = "[PRAÇA] "
        praca_to_lojas: Dict[str, List[str]] = {}
        praca_options: List[str] = []
        if indicadores_df is not None and not indicadores_df.empty:
            praca_col = _find_praca_col(indicadores_df)
            if praca_col and "Loja" in indicadores_df.columns:
                base_praca = indicadores_df[[praca_col, "Loja"]].copy()
                base_praca[praca_col] = base_praca[praca_col].astype(str).str.strip()
                base_praca["Loja"] = base_praca["Loja"].astype(str).str.strip()
                base_praca = base_praca[
                    (base_praca[praca_col] != "") & (base_praca["Loja"] != "")
                ]
                if not base_praca.empty:
                    for praca_name, group in base_praca.groupby(praca_col, dropna=False):
                        praca_label = str(praca_name).strip()
                        if not praca_label:
                            continue
                        lojas_lista = (
                            group["Loja"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .drop_duplicates()
                            .tolist()
                        )
                        if loja_opcoes:
                            lojas_lista = [loja for loja in lojas_lista if loja in loja_opcoes]
                        if lojas_lista:
                            praca_to_lojas[praca_label] = sorted(lojas_lista)
                if praca_to_lojas:
                    praca_options = [f"{praca_prefix}{p}" for p in sorted(praca_to_lojas.keys())]

        multiselect_options = [select_all_label] + praca_options + loja_opcoes
        def _expand_praca_selections(selections: List[str]) -> List[str]:
            if not selections or not praca_to_lojas:
                return selections
            expanded = list(selections)
            for opt in selections:
                if isinstance(opt, str) and opt.startswith(praca_prefix):
                    praca_name = opt[len(praca_prefix):].strip()
                    for loja in praca_to_lojas.get(praca_name, []):
                        if loja in loja_opcoes and loja not in expanded:
                            expanded.append(loja)
            return expanded
        def _on_change_lojas():
            selections = st.session_state.get("comparativo_lojas_sel", [])
            selections = [s for s in selections if s in multiselect_options]
            selections = _expand_praca_selections(selections)
            if select_all_label in selections:
                # se marcou somente "Selecionar todas", marcar todas
                if len(selections) == 1:
                    selections = multiselect_options[:]
                # se desmarcou alguma depois, remove a flag de selecionar todas
                elif len(selections) - 1 < len(loja_opcoes):
                    selections = [s for s in selections if s != select_all_label]
            selections = [s for s in selections if s in multiselect_options]
            selections = [s for s in multiselect_options if s in selections]
            st.session_state["comparativo_lojas_sel"] = selections
        st.multiselect(
            "Selecionar lojas",
            options=multiselect_options,
            key="comparativo_lojas_sel",
            on_change=_on_change_lojas,
            help="Escolha uma ou mais lojas ou selecione por praca.",
            placeholder="Clique para selecionar",
        )
        lojas_sel = _expand_praca_selections(
            [s for s in st.session_state.get("comparativo_lojas_sel", []) if s in multiselect_options]
        )

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
        indicadores_norm = _ensure_loja_key(indicadores_df) if indicadores_df is not None else None
        praca_col = _find_praca_col(indicadores_df) if indicadores_df is not None else None
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

        lojas_sel = _expand_praca_selections(
            [s for s in st.session_state.get("comparativo_lojas_sel", []) if s in multiselect_options]
        )
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
        def _warn_model_issue(bundle: Optional[Dict[str, object]], label: str) -> None:
            errors = (bundle or {}).get("errors") or {}
            msg = errors.get("catboost") or errors.get("_geral")
            if msg:
                st.warning(f"Modelo {label} indisponivel: {msg}")
        _warn_model_issue(model_hist, "historico")
        _warn_model_issue(model_ideal, "ideal")

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

            praca_val = None
            if indicadores_norm is not None and praca_col:
                praca_row, _ = _get_loja_row(indicadores_norm, loja_nome)
                if praca_row:
                    praca_val = str(praca_row.get(praca_col, "")).strip() or None
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
                    "Praca": praca_val,
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

        def _expand_axis(min_val: float, max_val: float) -> tuple[float, float]:
            if min_val == max_val:
                min_val -= 1
                max_val += 1
            span = max_val - min_val
            pad = max(1.0, span * 0.05)
            return float(min_val - pad), float(max_val + pad)

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
        st.markdown(
            """
            <style>
            div[data-testid="stExpander"] summary,
            div[data-testid="stExpander"] summary > div,
            div[data-testid="stExpander"] summary span,
            div[data-testid="stExpander"] summary p {
                font-size: 1.25rem !important;
                font-weight: 500 !important;
                line-height: 1.2 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for porte in ("Loja grande", "Loja média", "Loja pequena"):
            subset = tabela_comp.loc[tabela_comp["Porte"] == porte].reset_index(drop=True)
            count_label = f"{porte} ({len(subset)})"
            with st.expander(count_label, expanded=True):
                if subset.empty:
                    st.info("Sem registros.")
                    continue
                subset_exibir = subset.drop(columns=["Porte"], errors="ignore")
                subset_exibir = subset_exibir[[c for c in colunas_saida if c in subset_exibir.columns]]
                if "Diferença" in subset_exibir.columns:
                    subset_exibir = subset_exibir.copy()
                    urg_label = subset_exibir["Diferença"].apply(_delta_urgency_label)
                    subset_exibir["Urgência"] = urg_label.apply(lambda v: f"• {v}" if v else "")
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

        st.subheader("Grafico de Dimensionamento (Qtd Aux Real vs Ideal)")

        pontos_df = tabela_comp.copy()
        pontos_df["Qtd Aux Real Plot"] = pd.to_numeric(
            pontos_df["Qtd Aux Real"], errors="coerce"
        ).fillna(pd.to_numeric(pontos_df["Qtd Aux Historico"], errors="coerce"))
        pontos_df["Qtd Aux Ideal Plot"] = pd.to_numeric(
            pontos_df["Qtd Aux Ideal"], errors="coerce"
        )
        pontos_df["Faturamento/Qtd Aux Real (R$)"] = pontos_df[
            "Faturamento/Qtd Aux Real"
        ].apply(_format_brl)
        pontos_df["Praca"] = pontos_df["Praca"].fillna("").astype(str)
        pontos_df = pontos_df.dropna(subset=["Qtd Aux Real Plot", "Qtd Aux Ideal Plot"])

        if pontos_df.empty:
            st.info("Sem pontos validos para o grafico de comparativo.")
            return

        hist_vals = pd.to_numeric(pontos_df["Qtd Aux Historico"], errors="coerce")
        ideal_vals = pd.to_numeric(pontos_df["Qtd Aux Ideal"], errors="coerce")
        max_val = max(float(hist_vals.max(skipna=True)), float(ideal_vals.max(skipna=True)))
        if not math.isfinite(max_val):
            alt_max = pd.to_numeric(pontos_df["Qtd Aux Ideal Plot"], errors="coerce").max(skipna=True)
            max_val = float(alt_max) if pd.notna(alt_max) else 0.0
        step = 5
        upper = max(step, int(math.ceil(max_val / step) * step))
        if upper == int(max_val) and upper % step == 0:
            upper += step
        x_min, x_max = 0.0, float(upper)
        y_min, y_max = 0.0, float(upper)

        band_step = 0.25
        total_steps = int(round((x_max - x_min) / band_step))
        x_values = [x_min + (i * band_step) for i in range(total_steps + 1)]
        tick_values = list(range(0, upper + 1, step))

        offset_max = max(
            abs(y_max - x_min),
            abs(y_min - x_max),
            abs(y_max - x_max),
            abs(y_min - x_min),
            10,
        )

        band_defs = [
            {"label": "Otimo", "color": "#2c9a6c", "ranges": [(-1.5, 1.5)]},
            {"label": "Bom", "color": "#4da3f5", "ranges": [(1.5, 4.5), (-4.5, -1.5)]},
            {"label": "Atencao", "color": "#f0b429", "ranges": [(4.5, 9.5), (-9.5, -4.5)]},
            {"label": "Alto", "color": "#d8516d", "ranges": [(9.5, offset_max), (-offset_max, -9.5)]},
        ]

        band_layers: List[Dict[str, object]] = []
        for band in band_defs:
            for low, high in band["ranges"]:
                if high <= low:
                    continue
                band_layers.append(
                    {
                        "data": {
                            "sequence": {
                                "start": x_min,
                                "stop": x_max + band_step,
                                "step": band_step,
                                "as": "x",
                            }
                        },
                        "transform": [
                            {
                                "calculate": f"max({y_min}, datum.x + ({low}))",
                                "as": "y_lower",
                            },
                            {
                                "calculate": f"min({y_max}, datum.x + ({high}))",
                                "as": "y_upper",
                            },
                            {"filter": "datum.y_upper > datum.y_lower"},
                        ],
                        "mark": {"type": "area", "opacity": 0.15, "clip": True, "tooltip": None},
                        "encoding": {
                            "x": {
                                "field": "x",
                                "type": "quantitative",
                                "scale": {"domain": [x_min, x_max]},
                            },
                            "y": {
                                "field": "y_upper",
                                "type": "quantitative",
                                "scale": {"domain": [y_min, y_max]},
                            },
                            "y2": {"field": "y_lower"},
                            "color": {"value": band["color"]},
                        },
                    }
                )

        line_records = [{"x": x, "y": x} for x in x_values]

        x_mid = (x_min + x_max) / 2.0
        delta = max(1.5, (y_max - y_min) * 0.15)
        corner_pad = max(1.5, (y_max - y_min) * 0.06)
        upper_text = {
            "x": x_min + 1,
            "y": y_max - corner_pad,
            "label": "Dimensionamento\nacima do ideal",
        }
        lower_text = {
            "x": x_max - 1,
            "y": y_min + corner_pad + 2,
            "label": "Dimensionamento\nabaixo do ideal",
        }

        line_layer = {
            "data": {"values": line_records},
            "mark": {
                "type": "line",
                "color": "#2d2d2d",
                "strokeDash": [6, 4],
                "strokeWidth": 1,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "scale": {"domain": [x_min, x_max]},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "scale": {"domain": [y_min, y_max]},
                },
            },
        }
        upper_text_layer = {
            "data": {"values": [upper_text]},
            "mark": {
                "type": "text",
                "fontSize": 24,
                "fontWeight": "bold",
                "color": "#2d2d2d",
                "lineBreak": "\n",
                "align": "left",
                "baseline": "top",
                "tooltip": None,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "scale": {"domain": [x_min, x_max]},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "scale": {"domain": [y_min, y_max]},
                },
                "text": {"field": "label"},
            },
        }
        lower_text_layer = {
            "data": {"values": [lower_text]},
            "mark": {
                "type": "text",
                "fontSize": 24,
                "fontWeight": "bold",
                "color": "#2d2d2d",
                "lineBreak": "\n",
                "align": "right",
                "baseline": "bottom",
                "tooltip": None,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "scale": {"domain": [x_min, x_max]},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "scale": {"domain": [y_min, y_max]},
                },
                "text": {"field": "label"},
            },
        }
        points_layer = {
            "data": {"values": pontos_df.to_dict(orient="records")},
            "mark": {"type": "point", "filled": True, "size": 80, "color": "#1f2933"},
            "encoding": {
                "x": {
                    "field": "Qtd Aux Ideal Plot",
                    "type": "quantitative",
                    "scale": {"domain": [x_min, x_max]},
                    "axis": {
                        "title": "Qtd Aux Ideal",
                        "tickMinStep": 5,
                        "values": tick_values,
                        "format": ".0f",
                        "grid": True,
                    },
                },
                "y": {
                    "field": "Qtd Aux Real Plot",
                    "type": "quantitative",
                    "scale": {"domain": [y_min, y_max]},
                    "axis": {
                        "title": "Qtd Aux Real",
                        "tickMinStep": 5,
                        "values": tick_values,
                        "format": ".0f",
                        "grid": True,
                    },
                },
                "tooltip": [
                    {"field": "Loja", "type": "nominal", "title": "Loja"},
                    {"field": "Praca", "type": "nominal", "title": "Praca"},
                    {
                        "field": "Qtd Aux Historico",
                        "type": "quantitative",
                        "title": "Qtd Aux Historico",
                    },
                    {
                        "field": "Qtd Aux Ideal",
                        "type": "quantitative",
                        "title": "Qtd Aux Ideal",
                    },
                    {
                        "field": "Faturamento/Qtd Aux Real (R$)",
                        "type": "nominal",
                        "title": "Faturamento/Aux Real",
                    },
                ],
            },
        }

        chart_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": "container",
            "height": "container",
            "autosize": {"type": "fit", "contains": "padding"},
            "layer": band_layers + [line_layer, upper_text_layer, lower_text_layer, points_layer],
        }

        st.markdown(
            """
            <style>
            div[data-testid="stFullScreenFrame"] {
                overflow-x: hidden !important;
            }
            div[data-testid="stFullScreenFrame"] > div {
                display: flex !important;
                justify-content: center !important;
            }
            div[data-testid="stVegaLiteChart"] {
                width: 100% !important;
                max-width: 100% !important;
                height: calc(100vh - 150px) !important;
                max-height: calc(100vh - 150px) !important;
                padding-right: 12px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            }
            div[data-testid="stVegaLiteChart"] > div {
                width: 100% !important;
                height: 100% !important;
            }
            div[data-testid="stVegaLiteChart"] + div[data-testid="stVerticalBlock"] {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.vega_lite_chart(chart_spec, use_container_width=True)

        legend_html = """
        <div style="display:flex; gap:1.2rem; flex-wrap:wrap; align-items:center; justify-content:center; margin-top:0;">
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <span style="width:12px; height:12px; background:#2c9a6c; display:inline-block; border-radius:2px;"></span>
            <span>Otimo (<= 1)</span>
          </div>
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <span style="width:12px; height:12px; background:#4da3f5; display:inline-block; border-radius:2px;"></span>
            <span>Bom (2 a 4)</span>
          </div>
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <span style="width:12px; height:12px; background:#f0b429; display:inline-block; border-radius:2px;"></span>
            <span>Atencao (5 a 9)</span>
          </div>
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <span style="width:12px; height:12px; background:#d8516d; display:inline-block; border-radius:2px;"></span>
            <span>Alto (>= 10)</span>
          </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        st.subheader("Resumo")

        resumo_df = tabela_comp.copy()
        resumo_df["Diferença_num"] = pd.to_numeric(resumo_df["Diferença"], errors="coerce")
        resumo_df["Urgencia_label"] = resumo_df["Diferença_num"].apply(_delta_urgency_label)
        resumo_df["Urgencia_color"] = resumo_df["Urgencia_label"].apply(_delta_urgency_color)

        def _dimensionamento_bucket(diff_val: Optional[float]) -> str:
            if diff_val is None or (isinstance(diff_val, float) and pd.isna(diff_val)):
                return "Dimensionamento Ideal"
            if abs(diff_val) <= 1:
                return "Dimensionamento Ideal"
            return "Dimensionamento Abaixo do Ideal" if diff_val > 0 else "Dimensionamento Acima do Ideal"

        resumo_df["Dimensionamento"] = resumo_df["Diferença_num"].apply(_dimensionamento_bucket)

        faixa_labels = ["Ótimo", "Bom", "Atenção", "Alto"]
        faixa_colors = [
            _delta_urgency_color("Ótimo"),
            _delta_urgency_color("Bom"),
            _delta_urgency_color("Atenção"),
            _delta_urgency_color("Alto"),
        ]
        faixa_desc = [
            "Entre y=x-1 e y=x+1",
            "Entre y=x+2 e y=x+4 (e simétrico)",
            "Entre y=x+5 e y=x+9 (e simétrico)",
            "A partir de y=x+10 (e simétrico)",
        ]
        faixa_cols = st.columns(4)
        for idx, (label, color, desc) in enumerate(zip(faixa_labels, faixa_colors, faixa_desc)):
            count_val = int((resumo_df["Urgencia_label"] == label).sum())
            with faixa_cols[idx]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb; border-radius:8px; padding:0.75rem;">
                      <div style="display:flex; align-items:center; gap:0.5rem;">
                        <span style="width:12px; height:12px; background:{color}; display:inline-block; border-radius:2px;"></span>
                        <strong style="font-size:1rem;">{label}</strong>
                      </div>
                      <div style="color:#4b5563; font-size:0.9rem; margin-top:0.35rem;">{desc}</div>
                      <div style="font-size:1.4rem; font-weight:700; margin-top:0.35rem;">{count_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

        bucket_order = [
            "Dimensionamento Abaixo do Ideal",
            "Dimensionamento Ideal",
            "Dimensionamento Acima do Ideal",
        ]
        bucket_cols = st.columns(3)
        for idx, bucket in enumerate(bucket_order):
            bucket_df = resumo_df.loc[resumo_df["Dimensionamento"] == bucket].copy()
            total_diff = bucket_df["Diferença_num"].dropna().sum()
            lojas_list = bucket_df[
                ["Loja", "Urgencia_color", "Qtd Aux Ideal", "Qtd Aux Historico"]
            ].dropna(subset=["Loja"]).values.tolist()
            bullets = "".join(
                f"<li style='margin-bottom:0.25rem; list-style:none;'>"
                f"<span style='display:inline-block; width:8px; height:8px; background:{color}; border-radius:50%; margin-right:0.4rem;'></span>"
                f"{loja} <span style='color:#4b5563; font-size:0.85rem;'>(Ideal: {ideal}, Hist.: {hist})</span></li>"
                for loja, color, ideal, hist in lojas_list
            )
            with bucket_cols[idx]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb; border-radius:8px; padding:0.75rem; height:100%;">
                      <strong style="font-size:1rem;">{bucket}</strong>
                      <ul style="margin:0.6rem 0 0.4rem 0; padding:0;">{bullets or "<li style='list-style:none;'>Sem lojas</li>"}</ul>
                      <hr style="border:none; border-top:1px solid #e5e7eb; margin:0.6rem 0;" />
                      <div style="font-weight:700;">TOTAL: {int(total_diff) if pd.notna(total_diff) else 0}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
