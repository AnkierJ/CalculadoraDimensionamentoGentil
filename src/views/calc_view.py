import math
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.data.buscaDeLojas import _get_loja_row
from src.logic.core.logic import (
    DEFAULT_OCUPACAO_ALVO,
    DEFAULT_ABSENTEISMO,
    MODEL_ALGO_NAMES,
    _format_interval_value,
    _format_queue_diag,
    _metric_has_value,
    _train_cached,
    avaliar_carga_operacional_ideal,
    calcular_intervalos_modelos,
    calcular_resultado_ideal_simplificado,
    make_target,
    clean_training_dataframe,
    gerar_resultados_modelos,
    get_total_reference_values,
    montar_features_input,
    preparar_contexto_operacional,
    preparar_dicionarios_tempos_processos,
    preparar_indicadores_operacionais,
    prepare_training_dataframe,
)
from src.logic.utils.helpers import (
    _norm_code,
    _standardize_row,
    get_lookup,
    get_lookup_value,
    normalize_processo_nome,
    safe_float,
)
from src.logic.core.logic import apply_operacional_defaults_from_lookup


def render_calc_tab(tab_calc: DeltaGenerator) -> Dict[str, object]:
    """Renderiza a aba de cálculo até a preparação das features."""
    with tab_calc:
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


        def _compute_absenteismo_prefill(row_dict: Dict[str, object]) -> float:
            if not row_dict:
                return float(DEFAULT_ABSENTEISMO)
            disp_lookup = safe_float(get_lookup(row_dict, "%disp"), 0.0)
            absent_lookup = safe_float(get_lookup(row_dict, "%absent"), 0.0)
            if absent_lookup > 0:
                abs_val = absent_lookup if absent_lookup <= 1 else absent_lookup / 100.0
                return max(0.0, min(1.0, abs_val))
            if disp_lookup > 0:
                disp_val = disp_lookup if disp_lookup <= 1 else disp_lookup / 100.0
                return max(0.0, min(1.0, 1.0 - disp_val))
            return float(DEFAULT_ABSENTEISMO)


        # Pesquisa de loja
        st.markdown("**Pesquisar loja existente (opcional)**")
        col_lookup = st.columns([1, 1, 1])
        def _trigger_lookup_enter():
            st.session_state["lookup_enter_trigger"] = True
        with col_lookup[0]:
            lookup_field = st.radio(
                "",
                ["Loja", "SAP", "BCPS"],
                horizontal=True,
                key="lookup_field",
                label_visibility="collapsed",
            )

        df_ind = st.session_state.get("fIndicadores")
        df_estrutura = st.session_state.get("dEstrutura")
        df_pessoas = st.session_state.get("dPessoas")
        if lookup_field in ("BCPS", "SAP"):
            with col_lookup[1]:
                lookup_code = st.text_input(
                    "",
                    placeholder=(f"Código ({lookup_field})"),
                    key="lookup_code",
                    label_visibility="collapsed",
                    on_change=_trigger_lookup_enter,
                )
        elif lookup_field == "Loja":
            with col_lookup[1]:
                _ = st.text_input(
                    "",
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
                        st.session_state["absenteismo_input"] = _compute_absenteismo_prefill(st.session_state.get("lookup_row", {}))
                        apply_operacional_defaults_from_lookup(st.session_state["lookup_row"])
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
                    step=1.0,
                    format="%.1f",
                    help="Carga semanal prevista em contrato para cada auxiliar, antes de qualquer perda operacional.",
                )
                horas_disp = float(horas_disp_input)
                if horas_disp_input > 200:
                    horas_disp = horas_disp_input / 4.33
                    st.caption(f"Valor informado parece mensal. Convertido para {horas_disp:.1f} h/semana.")
                horas_loja_config_raw = safe_float(
                    st.session_state.get("horas_operacionais_form", st.session_state.get("horas_loja_config", 60.0)),
                    60.0,
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
            dias_operacionais_semana = int(st.session_state.get("dias_operacionais_loja_form", st.session_state.get("dias_operacionais_semana", 6)))
            dias_operacionais_semana = max(1, min(7, dias_operacionais_semana))
            if horas_loja_config <= 24:
                horas_loja_config = horas_loja_config * dias_operacionais_semana

            st.session_state["horas_disp_semanais"] = horas_disp
            st.session_state["horas_loja_config"] = horas_loja_config
            st.session_state["dias_operacionais_semana"] = dias_operacionais_semana
        else:
            horas_disp = 44.0
            horas_loja_config = float(st.session_state.get("horas_loja_config", 60.0))
            absenteismo = float(DEFAULT_ABSENTEISMO)
            folga_operacional = 0.15

        dias_operacionais_semana = int(st.session_state.get("dias_operacionais_loja_form", st.session_state.get("dias_operacionais_semana", 6)))
        dias_operacionais_semana = max(1, min(7, dias_operacionais_semana))
        st.session_state["dias_operacionais_semana"] = dias_operacionais_semana

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
                for key in ["Area Total", "Qtd Caixas", "HorasOperacionais", "DiasOperacionais"]:
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
            dias_operacionais_base = int(st.session_state.get("dias_operacionais_semana", 6))
            dias_operacionais_em_uso = int(st.session_state.get("dias_operacionais_loja_form", dias_operacionais_base))
            dias_operacionais_em_uso = max(1, min(7, dias_operacionais_em_uso))
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
                dias_operacionais_prefill = int(estrutura_defaults.get("DiasOperacionais", dias_operacionais_em_uso))
                dias_operacionais_prefill = max(1, min(7, dias_operacionais_prefill))
                dias_operacionais_loja = st.number_input(
                    "Dias operacionais (dados da loja)",
                    min_value=1,
                    max_value=7,
                    step=1,
                    value=dias_operacionais_prefill,
                    help="Número de dias em que a loja mantém operação. Usado para converter horas diárias em semanais.",
                )
                dias_operacionais_loja = int(dias_operacionais_loja)
                dias_operacionais_em_uso = dias_operacionais_loja
                horas_op_default = float(estrutura_defaults.get("HorasOperacionais", 0.0) or st.session_state.get("horas_loja_config", 0.0))
                if horas_op_default > 0 and horas_op_default <= 24:
                    horas_op_default = horas_op_default * dias_operacionais_em_uso
                horas_operacionais_input = st.number_input(
                    "Horas operacionais (h/semana)",
                    min_value=7.0,
                    max_value=168.0,
                    step=1.0,
                    value=float(horas_op_default),
                    format="%.0f",
                    help="Tempo total semanal de funcionamento da loja (ex: 60h para 10h/dia x 6 dias). Alimenta os cálculos ideais/ML.",
                )
                horas_operacionais_semanais = float(horas_operacionais_input)
                horas_operacionais_diarias = horas_operacionais_semanais / max(1, dias_operacionais_em_uso)
            st.session_state["dias_operacionais_loja_form"] = dias_operacionais_em_uso
            st.session_state["dias_operacionais_semana"] = dias_operacionais_em_uso

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
            indicadores_ctx = preparar_indicadores_operacionais(
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
            pct_base_total = indicadores_ctx["pct_base_total"]
            pct_faturamento = indicadores_ctx["pct_faturamento"]
            pct_ativos = indicadores_ctx["pct_ativos"]
            taxa_inicios = indicadores_ctx["taxa_inicios"]
            taxa_reativacao = indicadores_ctx["taxa_reativacao"]
            taxa_reinicio = indicadores_ctx["taxa_reinicio"]
            cluster_values = indicadores_ctx["cluster_values"]
            cluster_result = indicadores_ctx["cluster_result"]
            cluster_used = indicadores_ctx["cluster_used"]
            for msg_type, msg_text in indicadores_ctx["messages"]:
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

            op_caption = "Indicadores operacionais (dados históricos da loja selecionada)" if has_lookup else "Indicadores operacionais estimados por clusterização"
            with st.expander(op_caption):
                colFlow1, colFlow2, colFlow3 = st.columns(3)
                with colFlow1:
                    st.metric("Pedidos/Hora", f"{cluster_values['Pedidos/Hora']:.2f}")
                    st.metric("Pedidos/Dia", f"{cluster_values['Pedidos/Dia']:.2f}")
                with colFlow2:
                    st.metric("Itens/Pedido", f"{cluster_values['Itens/Pedido']:.2f}")
                    st.metric("Faturamento/Hora", f"R$ {cluster_values['Faturamento/Hora']:.2f}")
                with colFlow3:
                    st.metric("% Retirada", f"{cluster_values['%Retirada']:.2f}%")
                if cluster_used and cluster_result:
                    cluster_id = int(cluster_result.get("cluster_id", 0)) + 1
                    st.info(
                        f"Indicadores estimados via clusterização histórica (cluster {cluster_id}/{cluster_result.get('n_clusters')} com {cluster_result.get('cluster_size')} lojas)."
                    )

            pedidos_hora = cluster_values["Pedidos/Hora"]
            pedidos_dia = cluster_values["Pedidos/Dia"]
            itens_pedido = cluster_values["Itens/Pedido"]
            faturamento_hora = cluster_values["Faturamento/Hora"]
            pct_retirada = cluster_values["%Retirada"]

            # Derivar faturamento/hora a partir de Receita Total / mês, dias operacionais e horas operacionais
            denom_fat_hora = 4.34 * max(1.0, float(dias_operacionais_em_uso)) * max(0.1, float(horas_operacionais_diarias))
            if receita_total > 0 and denom_fat_hora > 0:
                faturamento_hora_calc = float(receita_total) / denom_fat_hora
                if faturamento_hora_calc > 0:
                    faturamento_hora = faturamento_hora_calc
                    cluster_values["Faturamento/Hora"] = faturamento_hora_calc

            st.session_state["horas_operacionais_form"] = float(horas_operacionais_semanais)

            features_input = montar_features_input(
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
        with st.form("form_inputs"):
            st.markdown(
                """
                <style>
                div[data-testid="stForm"] {
                    border: none !important;
                    box-shadow: none !important;
                    padding: 0 !important;
                }
                /* Destacar campos-chave de indicadores (container inteiro, incluindo botões +/-) */
                div[data-testid="stNumberInput"]:has(input[aria-label="Escritorio"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="Espaco Evento"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="Dias operacionais (dados da loja)"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="Horas operacionais (h/semana)"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="Base Ativa"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="Recuperados"]),
                div[data-testid="stNumberInput"]:has(input[aria-label="I4 a I6"]) {
                    border: 2px solid #0c0863 !important;
                    box-shadow: 0 0 0 3px rgba(12, 8, 99, 0.15) !important;
                    border-radius: 8px !important;
                    padding: 4px !important;
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

                # Não inferimos frequências automaticamente no modo simplificado; o usuário preenche todas.
                auto_freqs: Dict[str, float] = {}
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
                                freq_val = st.number_input(
                                    "Freq/semana",
                                    min_value=0.0,
                                    step=1.0,
                                    value=float(freq_default),
                                    format="%.1f",
                                    key=f"sim_proc_freq_{proc_norm}",
                                )
                            if usa_media_geral:
                                st.caption("Tempo vindo da média geral das lojas (sem dado específico da loja).")
                        updated_tempos[proc_norm] = tempo_val
                        updated_freqs[proc_norm] = freq_val
                st.session_state["sim_processos_tempos_custom"] = updated_tempos
                st.session_state["sim_processos_freq"] = updated_freqs
                st.session_state["sim_processos_auto_freq"] = auto_freqs
 
            mostrar_metricas = False
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if modo_ml:
                    # espaçamento para alinhar verticalmente com o slider ao lado
                    st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
                    mostrar_metricas = st.checkbox(
                        "Mostrar métricas/IC",
                        value=False,
                        help="Ative apenas se precisar das métricas e intervalos de confiança. Mantendo desativado o cálculo fica mais rápido.",
                    )
            with col2:

                if modo_ml:

                    anchor_percent = st.select_slider(

                        "Âncora receita/aux (%)",

                        options=[50, 55, 60, 65, 70, 75, 80, 85, 90],

                        value=int(st.session_state.get("anchor_rpa_percent", 60)),

                        help="Percentil de receita por auxiliar usado como referência: se a meta é evitar falta de gente, prefira percentil mais baixo; se a meta é eficiência agressiva, prefira percentil mais alto.",

                    )

                else:

                    anchor_percent = float(st.session_state.get("anchor_rpa_percent", 60))

        with col3:
            st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "Calcular Qtd Auxiliares",
                    type="primary",
                    use_container_width=True,
                )

    if submitted:
        st.session_state["anchor_rpa_percent"] = anchor_percent
        st.session_state["anchor_rpa_quantile"] = float(anchor_percent) / 100.0

    dias_operacionais_ativos = int(st.session_state.get("dias_operacionais_loja_form", dias_operacionais_semana))
    dias_operacionais_ativos = max(1, min(7, dias_operacionais_ativos))
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

    model_bundle_hist = None
    model_bundle_ideal = None
    if modo_ml:
        cache_ver = 9 + int(anchor_quantile * 100)
        model_bundle_hist = _train_cached(
            train_df,
            "historico",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver,
        )
        model_bundle_ideal = _train_cached(
            train_df,
            "ideal",
            horas_disp,
            margem,
            anchor_quantile=anchor_quantile,
            cache_version=cache_ver,
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
    )
    tmedio_min_atend = float(st.session_state.get("tmedio_min_atend", 6.0))
    result_ideal = None

    if modo_ml:
        resultados_modelos: List[Dict[str, object]] = []
        resultados_modelos_ideal: List[Dict[str, object]] = []
        model_errors_hist: Dict[str, object] = {}
        model_errors_ideal: Dict[str, object] = {}
        if model_bundle_hist is not None:
            resultados_modelos, model_errors_hist = gerar_resultados_modelos(
                model_bundle_hist,
                train_df,
                features_input,
                "historico",
                horas_disp,
                margem,
                anchor_quantile=anchor_quantile,
                compute_metrics=mostrar_metricas,
            )
            resultados_modelos = [res for res in resultados_modelos if res.get("key") == "catboost"]
        if model_bundle_ideal is not None:
            resultados_modelos_ideal, model_errors_ideal = gerar_resultados_modelos(
                model_bundle_ideal,
                train_df,
                features_input,
                "ideal",
                horas_disp,
                margem,
                anchor_quantile=anchor_quantile,
                compute_metrics=mostrar_metricas,
            )
            resultados_modelos_ideal = [res for res in resultados_modelos_ideal if res.get("key") == "catboost"]

        def _ajustar_pred_hist(pred_val: float) -> tuple[float, Dict[str, float]]:
            base_abs = float(absenteismo_prefill)
            base_folga = float(folga_base)
            fator_abs = (1.0 - base_abs) / max(0.1, 1.0 - float(absenteismo))
            fator_folga = (1.0 + float(folga_operacional)) / max(0.1, 1.0 + base_folga)
            ajuste_total = fator_abs * fator_folga
            return pred_val * ajuste_total, {
                "fator_abs": fator_abs,
                "fator_folga": fator_folga,
                "ajuste_total": ajuste_total,
            }

        def _ajustar_pred_ideal(pred_val: float) -> tuple[float, Dict[str, float]]:
            base_abs = float(absenteismo_prefill)
            base_folga = float(folga_base)
            fator_abs = (1.0 - base_abs) / max(0.1, 1.0 - float(absenteismo))
            fator_folga = (1.0 + float(folga_operacional)) / max(0.1, 1.0 + base_folga)
            ajuste_total = fator_abs * fator_folga
            return pred_val * ajuste_total, {
                "fator_abs": fator_abs,
                "fator_folga": fator_folga,
                "ajuste_total": ajuste_total,
            }

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
                features_input,
                "historico",
                horas_disp,
                margem,
                ["catboost"],
                anchor_quantile=anchor_quantile,
            ).get("catboost", {})
        if mostrar_metricas and cat_ideal:
            ci_ideal = calcular_intervalos_modelos(
                train_df,
                features_input,
                "ideal",
                horas_disp,
                margem,
                ["catboost"],
                anchor_quantile=anchor_quantile,
            ).get("catboost", {})

        if cat_hist and cat_ideal:
            st.success("Previsao (Machine Learning) concluida!")
            pred_hist_ajust, _ = _ajustar_pred_hist(float(cat_hist.get("pred") or 0.0))
            pred_ideal_ajust, _ = _ajustar_pred_ideal(float(cat_ideal.get("pred") or 0.0))
            pred_hist_int = int(round(pred_hist_ajust))
            pred_ideal_int = int(round(pred_ideal_ajust))
            diff_val = pred_ideal_ajust - pred_hist_ajust
            col_res = st.columns(3)
            with col_res[0]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Qtd Aux Histórico</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:600;\">{pred_hist_int} auxiliares</div>"
                    f"<div style=\"font-size:0.95rem;font-weight:400;color:#6c6c6c;\">{pred_hist_ajust:.2f} aux</div>"
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
                    f"<div style='font-size:1.3rem;font-weight:600;'>Qtd Aux Ideal</div>"
                    f"<div style='font-size:2.0rem;font-weight:700; line-height: 0.85;'>{pred_ideal_int} auxiliares</div>"
                    f"<div style='font-size:1.0rem;font-weight:400;color:#6c6c6c;'>({pred_ideal_ajust:.2f} aux)</div>"
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
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Diferenca (ideal - hist)</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:500;\">{diff_val:+.2f} auxiliares</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if mostrar_metricas:
                metrics_info_ideal = cat_ideal.get("metrics") or {}
                stats_parts: List[str] = []
                mape_v = metrics_info_ideal.get("MAPE")
                r2m = metrics_info_ideal.get("R2_mean")
                precisao = metrics_info_ideal.get("Precisao_percent")
                mae_v = metrics_info_ideal.get("MAE")
                rmse_v = metrics_info_ideal.get("RMSE")
                if _metric_has_value(precisao):
                    stats_parts.append(f"Precisao {precisao:.1f}%")
                if _metric_has_value(mape_v):
                    stats_parts.append(f"MAPE {mape_v*100:.1f}%")
                if _metric_has_value(mae_v):
                    stats_parts.append(f"MAE {mae_v:.2f}")
                if _metric_has_value(rmse_v):
                    stats_parts.append(f"RMSE {rmse_v:.2f}")
                if _metric_has_value(r2m):
                    stats_parts.append(f"R2 {r2m:.2f}")
                if stats_parts:
                    st.markdown(
                        f"<div style='text-align:center;'>Modelo de Machine Learning (ideal): {' | '.join(stats_parts)}</div>",
                        unsafe_allow_html=True,
                    )
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
            qtd_aux_atual = safe_float(get_lookup(lookup_row, "QtdAux"))

        ideal_val = float(result_ideal.get("qtd_aux_ideal", 0.0))
        if qtd_aux_atual is not None and not math.isnan(qtd_aux_atual):
            diff_val = ideal_val - float(qtd_aux_atual)
            col_res = st.columns(3)
            with col_res[0]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Qtd Aux Atual</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:500;\">{qtd_aux_atual:.2f} aux</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_res[1]:
                st.markdown(
                    f"<div style='text-align:center;color:#0c0863;background-color: #f0f2f6; border-radius: 10px; padding-bottom: 10px;'>"
                    f"<div style='font-size:1.3rem;font-weight:600;'>Qtd Aux Ideal</div>"
                    f"<div style='font-size:2.0rem;font-weight:600; line-height: 0.85;'>{ideal_val:.2f} aux</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_res[2]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style=\"font-size:1.1rem;font-weight:500;\">Diferenca (ideal - atual)</div>"
                    f"<div style=\"font-size:1.5rem;font-weight:500;\">{diff_val:+.2f} aux</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.metric("Qtd Auxiliares (ideal)", f"{ideal_val}")

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
# Helpers internos
from src.logic.models.model_fila import (
    estimate_queue_inputs,
    calcular_fila,
    QUEUE_CALIBRATION_DEFAULT,
)


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
    fila_text = f"Headcount (fila): {c_fila:.2f} auxiliares"
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
        line = f"{label}: {c_pred} auxiliares"
        extras: List[str] = []
        if _metric_has_value(rho_val):
            extras.append(f"ρ ≈ {rho_val:.2f}")
        if c_fila > 0:
            delta_pct = delta_abs / c_fila
            extras.append(f"Δ vs fila: {delta_abs:+.2f} aux ({delta_pct*100:+.1f}%)")
        elif delta_abs != 0:
            extras.append(f"Δ vs fila: {delta_abs:+.2f} aux")
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
