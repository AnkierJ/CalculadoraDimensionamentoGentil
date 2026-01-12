# =============================================================================
# Imports
# =============================================================================
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.core.logic import (
    clean_training_dataframe,
    get_schema_dAmostras,
    get_schema_dEstrutura,
    get_schema_dPessoas,
    get_schema_fFaturamento2,
    get_schema_fIndicadores,
    prepare_training_dataframe,
    render_append,
    _assign_porte_cluster,
    _compute_porte_cluster_context,
    _find_praca_col,
    _is_loja_grande,
)
from src.logic.data.buscaDeLojas import _ensure_loja_key


# =============================================================================
# Render
# =============================================================================
def render_dados_tab(tab_dados: DeltaGenerator, paths: Dict[str, Path]) -> None:
    with tab_dados:
        st.subheader("Dados de base")
        # =============================================================================
        # Preview / Upload
        # =============================================================================
        aba_preview, aba_upload, aba_criterios = st.tabs(
            ["Base atual (somente leitura)", "Adicionar dados (upload)", "Critérios"]
        )
        # =============================================================================
        # Preview
        # =============================================================================
        with aba_preview:
            st.caption("Pré-visualização das 10 primeiras linhas das tabelas carregadas por padrão do diretório 'data/'.")
            with st.expander("dAmostras"):
                st.dataframe(st.session_state["dAmostras"].head(10), use_container_width=True)
                path = paths.get("dAmostras")
                if path and path.exists():
                    st.write(f"📦 Tamanho de dAmostras.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo dAmostras.csv não encontrado.")
            with st.expander("dEstrutura"):
                st.dataframe(st.session_state["dEstrutura"].head(10), use_container_width=True)
                path = paths.get("dEstrutura")
                if path and path.exists():
                    st.write(f"📦 Tamanho de dEstrutura.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo dEstrutura.csv não encontrado.")
            with st.expander("dPessoas"):
                st.dataframe(st.session_state["dPessoas"].head(10), use_container_width=True)
                path = paths.get("dPessoas")
                if path and path.exists():
                    st.write(f"📦 Tamanho de dPessoas.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo dPessoas.csv não encontrado.")
            with st.expander("fFaturamento2"):
                st.dataframe(st.session_state["fFaturamento2"].head(10), use_container_width=True)
                path = paths.get("fFaturamento2")
                if path and path.exists():
                    st.write(f"📦 Tamanho de fFaturamento2.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo fFaturamento2.csv não encontrado.")
            with st.expander("fIndicadores"):
                st.dataframe(st.session_state["fIndicadores"].head(10), use_container_width=True)
                path = paths.get("fIndicadores")
                if path and path.exists():
                    st.write(f"📦 Tamanho de fIndicadores.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo fIndicadores.csv não encontrado.")
            st.divider()
            colx1, colx2, colx3 = st.columns([1, 3, 1], gap="medium", vertical_alignment="center")
            with colx2:
                if st.button("Gerar arquivos para download", use_container_width=True, type="primary"):
                    st.session_state["downloads_ready"] = True

            if st.session_state.get("downloads_ready"):
                for nome in ["dAmostras", "dEstrutura", "dPessoas", "fFaturamento2", "fIndicadores"]:
                    path = paths.get(nome)
                    if path and path.exists():
                        csv_bytes = path.read_bytes()
                    else:
                        df = st.session_state[nome]
                        csv_bytes = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                    st.download_button(
                        label=f"⬇️ Baixar {nome}.csv",
                        data=csv_bytes,
                        file_name=f"{nome}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        # =============================================================================
        # Upload
        # =============================================================================
        with aba_upload:
            st.caption("Envie arquivos para ACRESCENTAR dados à base atual. Entradas duplicadas são deduplicadas por chaves básicas.")
            tabs = st.tabs(["dAmostras", "dEstrutura", "dPessoas", "fFaturamento2", "fIndicadores"])
            with tabs[0]:
                render_append("dAmostras", get_schema_dAmostras, ["Loja", "Processo", "Amostra"])
            with tabs[1]:
                render_append("dEstrutura", get_schema_dEstrutura, ["Loja"])
            with tabs[2]:
                render_append("dPessoas", get_schema_dPessoas, ["Loja"])
            with tabs[3]:
                render_append("fFaturamento2", get_schema_fFaturamento2, ["Loja", "CodPedido"])
            with tabs[4]:
                render_append("fIndicadores", get_schema_fIndicadores, ["Loja"])

        # =============================================================================
        # Critérios
        # =============================================================================
        with aba_criterios:
            st.markdown(
                """
                <style>
                .criteria-rankings [data-testid="stDataFrame"] {
                    overflow: visible !important;
                }
                .criteria-rankings [data-testid="stDataFrame"] > div {
                    overflow: visible !important;
                }
                .criteria-rankings [data-testid="stDataFrame"] div[role="grid"] {
                    overflow-x: hidden !important;
                    overflow-y: auto !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Rankings de faturamento por TotalMapeado, SalarioMapeado e SalarioMapeado*%IAF25.")
            estrutura_df = st.session_state.get("dEstrutura")
            pessoas_df = st.session_state.get("dPessoas")
            indicadores_df = st.session_state.get("fIndicadores")

            if estrutura_df is None or estrutura_df.empty:
                st.warning("Base de estrutura vazia ou nÇœo carregada.")
                return
            if pessoas_df is None or pessoas_df.empty:
                st.warning("Base de pessoas vazia ou nÇœo carregada.")
                return

            train_df = prepare_training_dataframe(estrutura_df, pessoas_df, indicadores_df)
            train_df = clean_training_dataframe(train_df)
            if train_df is None or train_df.empty:
                st.warning("Sem dados vÇ­lidos para gerar os rankings (verifique dEstrutura/dPessoas/fIndicadores).")
                return

            train_norm = _ensure_loja_key(train_df)
            praca_col = _find_praca_col(indicadores_df) if indicadores_df is not None else None
            if indicadores_df is not None and not indicadores_df.empty and praca_col:
                indicadores_norm = _ensure_loja_key(indicadores_df)
                base_praca = indicadores_norm[[praca_col, "Loja_norm"]].copy()
                base_praca[praca_col] = base_praca[praca_col].astype(str).str.strip()
                base_praca = base_praca[base_praca[praca_col] != ""]
                base_praca = base_praca.dropna(subset=["Loja_norm"]).drop_duplicates("Loja_norm")
                train_norm = train_norm.merge(base_praca, on="Loja_norm", how="left")

            receita = (
                pd.to_numeric(train_norm["ReceitaTotalMes"], errors="coerce")
                if "ReceitaTotalMes" in train_norm.columns
                else pd.Series(pd.NA, index=train_norm.index, dtype="float64")
            )
            total_map = (
                pd.to_numeric(train_norm["TotalMapeado"], errors="coerce")
                if "TotalMapeado" in train_norm.columns
                else pd.Series(pd.NA, index=train_norm.index, dtype="float64")
            )
            salario_map = (
                pd.to_numeric(train_norm["SalarioMapeado"], errors="coerce")
                if "SalarioMapeado" in train_norm.columns
                else pd.Series(pd.NA, index=train_norm.index, dtype="float64")
            )
            iaf_25 = (
                pd.to_numeric(train_norm["%IAF25"], errors="coerce")
                if "%IAF25" in train_norm.columns
                else pd.Series(pd.NA, index=train_norm.index, dtype="float64")
            )
            qtd_aux = (
                pd.to_numeric(train_norm["QtdAux"], errors="coerce")
                if "QtdAux" in train_norm.columns
                else pd.Series(pd.NA, index=train_norm.index, dtype="float64")
            )
            qtd_aux_real = total_map.where(total_map.notna() & (total_map > 0), qtd_aux)

            ratio_total = receita / total_map.replace(0, pd.NA)
            salario_fallback = salario_map.where(salario_map.notna() & (salario_map > 0), total_map)
            ratio_salario = receita / salario_fallback.replace(0, pd.NA)
            iaf_norm = iaf_25.copy()
            if not iaf_norm.empty:
                iaf_norm = iaf_norm.where(iaf_norm <= 1.5, iaf_norm / 100.0)
            ratio_salario_iaf = ratio_salario * iaf_norm.fillna(1.0)

            horas_disp = float(st.session_state.get("horas_disp_semanais", 44.0))
            margem = float(st.session_state.get("folga_operacional", 0.15))
            anchor_quantile = float(st.session_state.get("anchor_rpa_quantile", 0.5))
            cluster_ctx = _compute_porte_cluster_context(
                train_df,
                mode="historico",
                horas_disp=horas_disp,
                margem=margem,
                anchor_quantile=anchor_quantile,
            )

            def _classificar_porte(row: pd.Series) -> str:
                if not cluster_ctx:
                    return "Media"
                row_dict = row.to_dict()
                porte_map = cluster_ctx.get("porte_map", {}) or {}
                cid, _, _ = _assign_porte_cluster(row_dict, cluster_ctx)
                porte_code = porte_map.get(cid)
                thresholds = cluster_ctx.get("thresholds", {}) or {}
                ratios = []
                for col, thr in thresholds.items():
                    try:
                        val = float(row_dict.get(col, 0.0))
                    except Exception:
                        val = 0.0
                    if thr and thr > 0 and val >= 0:
                        ratios.append(val / thr)
                max_ratio = max(ratios) if ratios else 0.0
                if porte_code is not None:
                    if porte_code == 1:
                        return "Grande"
                    if porte_code in (2, 3):
                        return "Media"
                    return "Pequena"
                is_large = _is_loja_grande(row_dict, thresholds)
                if is_large or max_ratio >= 1.0:
                    return "Grande"
                if max_ratio <= 0.5:
                    return "Pequena"
                return "Media"

            porte_series = train_norm.apply(_classificar_porte, axis=1)
            praca_series = (
                train_norm[praca_col].fillna("").astype(str).str.strip()
                if praca_col and praca_col in train_norm.columns
                else pd.Series("", index=train_norm.index, dtype="object")
            )
            praca_series = praca_series.replace("", "Sem praca")

            base_df = pd.DataFrame(
                {
                    "Loja": train_norm["Loja"].astype(str).str.strip(),
                    "Qtd Aux Real": qtd_aux_real,
                    "Porte": porte_series,
                    "Praca": praca_series,
                    "Faturamento/TotalMap": ratio_total,
                    "Faturamento/SalarioMap": ratio_salario,
                    "Faturamento/SalarioMap*IAF25": ratio_salario_iaf,
                }
            )

            porte_opts = ["Todos"] + sorted(base_df["Porte"].dropna().unique().tolist())
            praca_opts = ["Todas"] + sorted(base_df["Praca"].dropna().unique().tolist())
            col_filt_1, col_filt_2 = st.columns(2)
            with col_filt_1:
                porte_sel = st.selectbox("Filtrar por porte", options=porte_opts)
            with col_filt_2:
                praca_sel = st.selectbox("Filtrar por praca", options=praca_opts)

            filtered_df = base_df.copy()
            if porte_sel != "Todos":
                filtered_df = filtered_df.loc[filtered_df["Porte"] == porte_sel]
            if praca_sel != "Todas":
                filtered_df = filtered_df.loc[filtered_df["Praca"] == praca_sel]

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

            def _format_ratio(val: Optional[float]) -> str:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return ""
                try:
                    num = float(val)
                except Exception:
                    return ""
                formatted = f"{num:,.2f}"
                return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

            def _format_aux(val: Optional[float]) -> str:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "-"
                try:
                    num = float(val)
                except Exception:
                    return "-"
                return str(int(round(num)))

            def _hex_to_rgb(color: str) -> tuple[int, int, int]:
                color = color.lstrip("#")
                return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

            def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
                return "#{:02x}{:02x}{:02x}".format(*rgb)

            def _lerp_color(low: str, high: str, t: float) -> str:
                t = max(0.0, min(1.0, t))
                r1, g1, b1 = _hex_to_rgb(low)
                r2, g2, b2 = _hex_to_rgb(high)
                r = int(r1 + (r2 - r1) * t)
                g = int(g1 + (g2 - g1) * t)
                b = int(b1 + (b2 - b1) * t)
                return _rgb_to_hex((r, g, b))

            def _gradient_color(t: float) -> str:
                # Low -> red, mid -> yellow, high -> green
                if t <= 0.5:
                    return _lerp_color("#d8516d", "#f0b429", t / 0.5)
                return _lerp_color("#f0b429", "#2c9a6c", (t - 0.5) / 0.5)

            def _build_rank_dataframe(
                df: pd.DataFrame, ratio_col: str, ratio_label: str, use_brl: bool
            ) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
                if df.empty or ratio_col not in df.columns:
                    return None, None
                data = df.dropna(subset=[ratio_col]).copy()
                data = data[pd.to_numeric(data[ratio_col], errors="coerce").notna()]
                data = data.sort_values(ratio_col, ascending=False)
                if data.empty:
                    return None, None
                ratio_vals = pd.to_numeric(data[ratio_col], errors="coerce")
                min_val = float(ratio_vals.min(skipna=True))
                max_val = float(ratio_vals.max(skipna=True))
                if not math.isfinite(min_val) or not math.isfinite(max_val):
                    min_val, max_val = 0.0, 0.0
                denom = max_val - min_val
                rows: List[Dict[str, object]] = []
                colors: List[str] = []
                for _, row in data.iterrows():
                    val = row.get(ratio_col)
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    pct = 0.5 if denom == 0 else (float(val) - min_val) / denom
                    color = _gradient_color(pct)
                    rows.append(
                        {
                            "Indicador": "",
                            "Loja": str(row.get("Loja", "")).strip(),
                            "Time Comercial (TOTAL MAP)": _format_aux(row.get("Qtd Aux Real")),
                            "Porte": str(row.get("Porte", "")).strip(),
                            "Praca": str(row.get("Praca", "")).strip(),
                            ratio_label: _format_brl(val) if use_brl else _format_ratio(val),
                        }
                    )
                    colors.append(color)
                rank_df = pd.DataFrame(rows)
                if rank_df.empty:
                    return None, None
                color_series = pd.Series(colors, index=rank_df.index)
                return rank_df, color_series

            st.markdown('<div class="criteria-rankings">', unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Ranking Faturamento/TotalMap")
                rank_df, color_series = _build_rank_dataframe(
                    filtered_df,
                    ratio_col="Faturamento/TotalMap",
                    ratio_label="Faturamento/TotalMap",
                    use_brl=True,
                )
                if rank_df is None or rank_df.empty:
                    st.info("Sem registros.")
                else:
                    def _style_indicator(row: pd.Series) -> List[str]:
                        color = "#e5e7eb"
                        if color_series is not None and row.name in color_series.index:
                            color = color_series.loc[row.name]
                        styles = []
                        for col in row.index:
                            if col == "Indicador":
                                styles.append(
                                    f"background-color: {color}; border-radius: 6px;"
                                )
                            else:
                                styles.append("")
                        return styles
                    styled = rank_df.style.apply(_style_indicator, axis=1)
                    st.dataframe(
                        styled,
                        use_container_width=True,
                        height=520,
                        column_config={
                            "Indicador": st.column_config.Column(label="Indicador", width="small"),
                            "Time Comercial (TOTAL MAP)": st.column_config.Column(width="small"),
                            "Porte": st.column_config.Column(width="small"),
                            "Praca": st.column_config.Column(width="small"),
                        },
                    )
            with col_right:
                st.markdown("#### Ranking Faturamento/SalarioMap")
                rank_df, color_series = _build_rank_dataframe(
                    filtered_df,
                    ratio_col="Faturamento/SalarioMap",
                    ratio_label="Faturamento/SalarioMap",
                    use_brl=False,
                )
                if rank_df is None or rank_df.empty:
                    st.info("Sem registros.")
                else:
                    def _style_indicator(row: pd.Series) -> List[str]:
                        color = "#e5e7eb"
                        if color_series is not None and row.name in color_series.index:
                            color = color_series.loc[row.name]
                        styles = []
                        for col in row.index:
                            if col == "Indicador":
                                styles.append(
                                    f"background-color: {color}; border-radius: 6px;"
                                )
                            else:
                                styles.append("")
                        return styles
                    styled = rank_df.style.apply(_style_indicator, axis=1)
                    st.dataframe(
                        styled,
                        use_container_width=True,
                        height=520,
                        column_config={
                            "Indicador": st.column_config.Column(label="Indicador", width="small"),
                            "Time Comercial (TOTAL MAP)": st.column_config.Column(width="small"),
                            "Porte": st.column_config.Column(width="small"),
                            "Praca": st.column_config.Column(width="small"),
                        },
                    )
            st.markdown("#### Ranking Faturamento/SalarioMap*%IAF25")
            rank_df, color_series = _build_rank_dataframe(
                filtered_df,
                ratio_col="Faturamento/SalarioMap*IAF25",
                ratio_label="Faturamento/SalarioMap*%IAF25",
                use_brl=False,
            )
            if rank_df is None or rank_df.empty:
                st.info("Sem registros.")
            else:
                def _style_indicator(row: pd.Series) -> List[str]:
                    color = "#e5e7eb"
                    if color_series is not None and row.name in color_series.index:
                        color = color_series.loc[row.name]
                    styles = []
                    for col in row.index:
                        if col == "Indicador":
                            styles.append(
                                f"background-color: {color}; border-radius: 6px;"
                            )
                        else:
                            styles.append("")
                    return styles
                styled = rank_df.style.apply(_style_indicator, axis=1)
                st.dataframe(
                    styled,
                    use_container_width=True,
                    height=520,
                    column_config={
                        "Indicador": st.column_config.Column(label="Indicador", width="small"),
                        "Time Comercial (TOTAL MAP)": st.column_config.Column(width="small"),
                        "Porte": st.column_config.Column(width="small"),
                        "Praca": st.column_config.Column(width="small"),
                    },
                )
            st.markdown("</div>", unsafe_allow_html=True)
