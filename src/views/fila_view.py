import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.data.buscaDeLojas import _filter_df_by_loja
from src.logic.core.logic import (
    WEEKS_PER_MONTH,
    agregar_tempo_medio_por_processo,
    inferir_frequencia_por_processo,
)
from src.logic.models.model_fila import (
    DEFAULT_OCUPACAO_ALVO,
    QUEUE_CALIBRATION_DEFAULT,
    calcular_fila,
)


def render_fila_tab(tab_fila: DeltaGenerator) -> None:
    with tab_fila:
        st.subheader("Simulador - Teoria das filas")
        amostras_df = st.session_state.get("dAmostras")
        if amostras_df is None or amostras_df.empty:
            st.warning("Carregue dAmostras para usar esta aba.")
            return

        lookup_row = st.session_state.get("lookup_row", {}) or {}
        loja_default = str(lookup_row.get("Loja", "") or "").strip()
        lojas_amostras = [
            str(loja).strip()
            for loja in amostras_df["Loja"].dropna().unique().tolist()
            if str(loja).strip()
        ]
        lojas_amostras = sorted(set(lojas_amostras))
        opcoes_loja = ["Média geral"] + lojas_amostras
        idx_default = opcoes_loja.index(loja_default) if loja_default in opcoes_loja else 0
        loja_escolhida = st.selectbox(
            "Use os tempos médios de qual loja?",
            opcoes_loja,
            index=idx_default,
            help="Pré-preenche tempos e frequências com a média calculada em dAmostras.",
        )

        amostras_base = amostras_df
        if loja_escolhida != "Média geral":
            amostras_loja, _ = _filter_df_by_loja(amostras_df, loja_escolhida)
            if not amostras_loja.empty:
                amostras_base = amostras_loja
            else:
                st.warning("Não encontrei amostras para essa loja; usando média geral.")

        tempos_df = agregar_tempo_medio_por_processo(amostras_base)
        freq_df = inferir_frequencia_por_processo(amostras_base)
        if tempos_df.empty:
            st.warning("Sem tempos médios calculados em dAmostras.")
            return

        merge_base = tempos_df[["Processo", "tempo_medio_min"]].copy()
        freq_col = (
            freq_df[["Processo", "frequencia"]]
            if freq_df is not None and not freq_df.empty
            else pd.DataFrame(columns=["Processo", "frequencia"])
        )
        processos_df = pd.merge(merge_base, freq_col, on="Processo", how="outer")
        if processos_df.empty:
            st.warning("Nenhum processo encontrado em dAmostras.")
            return

        processos_df["tempo_medio_min"] = pd.to_numeric(processos_df["tempo_medio_min"], errors="coerce").fillna(0.0)
        processos_df["frequencia"] = pd.to_numeric(processos_df["frequencia"], errors="coerce").fillna(0.0)
        processos_df = (
            processos_df.groupby("Processo", dropna=False)
            .agg({"tempo_medio_min": "mean", "frequencia": "mean"})
            .reset_index()
        )
        processos_df["Processo"] = processos_df["Processo"].fillna("Processo sem nome")

        st.caption("Edite livremente tempos e frequências; valores iniciais vêm de dAmostras.")
        with st.form("fila_livre_form"):
            submitted_fila = False
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                freq_periodo = st.selectbox(
                    "Frequência informada",
                    ["por semana", "por dia", "por mês"],
                    help="Define o período usado para os campos de frequência abaixo.",
                )
                dias_oper = st.number_input(
                    "Dias operacionais",
                    min_value=1,
                    max_value=7,
                    value=int(st.session_state.get("dias_operacionais_loja_form", st.session_state.get("dias_operacionais_semana", 6))),
                    help="Usado para converter frequências diárias em semana e lambda/hora.",
                )
                horas_dia_default_raw = float(st.session_state.get("horas_operacionais_form", st.session_state.get("horas_loja_config", 10.0)))
                if horas_dia_default_raw > 24:
                    horas_dia_default_raw = horas_dia_default_raw / max(float(dias_oper), 1.0)
                horas_dia_default = max(1.0, min(24.0, horas_dia_default_raw))
                horas_dia = st.number_input(
                    "Horas operacionais por dia",
                    min_value=1.0,
                    max_value=24.0,
                    step=0.5,
                    value=horas_dia_default,
                    format="%.1f",
                    help="Horas de funcionamento da loja consideradas para calcular lambda.",
                )
            with col_cfg2:
                horas_contratuais = st.number_input(
                    "Horas contratuais por auxiliar (semana)",
                    min_value=1.0,
                    step=1.0,
                    value=44.0,
                    format="%.1f",
                    help="Usado para ajustar headcount quando a carga semanal da loja excede essa jornada.",
                )
                absenteismo_pct = st.number_input(
                    "Absenteísmo (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    value=DEFAULT_OCUPACAO_ALVO * 100,
                    format="%.1f",
                    help="Reduz as horas produtivas disponíveis por auxiliar.",
                )
                folga_extra_pct = st.number_input(
                    "Folga extra/SLA (%)",
                    min_value=0.0,
                    max_value=200.0,
                    step=1.0,
                    value=0.0,
                    format="%.1f",
                    help="Reserva adicional sobre o headcount calculado.",
                )
            with col_cfg3:
                rho_target = st.number_input(
                    "Ocupação alvo (rho)",
                    min_value=0.05,
                    max_value=0.99,
                    step=0.01,
                    value=DEFAULT_OCUPACAO_ALVO,
                    format="%.2f",
                    help="Controle de utilização máxima desejada por auxiliar.",
                )
                calibration_factor = st.number_input(
                    "Fator de calibração (carga)",
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    value=QUEUE_CALIBRATION_DEFAULT,
                    format="%.2f",
                    help="Multiplica a carga de chegada antes de aplicar a fórmula da fila.",
                )

            st.markdown("#### Processos e frequências")
            horas_semana_oper = horas_dia * dias_oper
            freq_label = {
                "por semana": "freq/semana",
                "por dia": "freq/dia",
                "por mês": "freq/mês",
            }[freq_periodo]

            state_key = f"fila_inputs_{loja_escolhida or 'geral'}"
            state_map = st.session_state.get(state_key, {})
            processos_resultados = []
            for _, proc_row in processos_df.sort_values("Processo").iterrows():
                proc_nome = str(proc_row["Processo"]).strip() or "Processo"
                proc_norm = proc_nome.lower().strip()
                defaults = state_map.get(proc_norm, {})
                tempo_val = st.number_input(
                    f"{proc_nome} - tempo médio (min)",
                    min_value=0.0,
                    value=float(defaults.get("tempo", proc_row["tempo_medio_min"] or 0.0)),
                    step=0.5,
                    format="%.2f",
                    key=f"fila_tempo_{state_key}_{proc_norm}",
                )
                freq_val = st.number_input(
                    f"{proc_nome} - {freq_label}",
                    min_value=0.0,
                    value=float(defaults.get("freq", proc_row["frequencia"] or 0.0)),
                    step=0.5,
                    format="%.2f",
                    key=f"fila_freq_{state_key}_{proc_norm}",
                )
                state_map[proc_norm] = {"tempo": tempo_val, "freq": freq_val}

                freq_semana = freq_val
                if freq_periodo == "por dia":
                    freq_semana = freq_val * dias_oper
                elif freq_periodo == "por mês":
                    freq_semana = freq_val / WEEKS_PER_MONTH
                lambda_proc = freq_semana / max(horas_semana_oper, 1e-6)
                carga_horas_semana = (freq_semana * tempo_val) / 60.0
                processos_resultados.append(
                    {
                        "Processo": proc_nome,
                        "tempo_min": tempo_val,
                        "freq_informada": freq_val,
                        "freq_semana": freq_semana,
                        "lambda_hora": lambda_proc,
                        "carga_horas_semana": carga_horas_semana,
                    }
                )

            st.session_state[state_key] = state_map
            submitted_fila = st.form_submit_button(
                "Calcular fila livre",
                type="primary",
                use_container_width=True,
            )

        if not submitted_fila:
            return

        ativos = [p for p in processos_resultados if p["lambda_hora"] > 0 and p["tempo_min"] > 0]
        lambda_total = sum(p["lambda_hora"] for p in ativos)
        lambda_tempo = sum(p["lambda_hora"] * p["tempo_min"] for p in ativos)
        if lambda_total <= 0 or lambda_tempo <= 0:
            st.warning("Informe frequências > 0 para calcular a fila.")
            return

        tma_min = lambda_tempo / lambda_total
        fila_diag = calcular_fila(
            lambda_total,
            tma_min,
            rho_target=rho_target,
            calibration_factor=calibration_factor,
        )
        horas_produtivas = horas_contratuais * (1 - absenteismo_pct / 100)
        horas_produtivas = max(horas_produtivas, 1e-3)
        cobertura = horas_semana_oper / horas_produtivas if horas_semana_oper > 0 else 1.0
        cobertura = max(cobertura, 1.0)
        c_base = float(fila_diag.get("c_fila", 0.0) or 0.0)
        c_ajust = c_base * cobertura
        c_final = c_ajust * (1 + folga_extra_pct / 100)

        st.success("Resultado da fila")
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Lambda total (chegadas/h)", f"{lambda_total:.2f}")
        with col_res2:
            st.metric("TMA ponderado (min)", f"{tma_min:.2f}")
        with col_res3:
            st.metric("Headcount bruto (fila)", f"{c_base:.2f}")
        st.metric("Headcount ajustado (absenteísmo/horas)", f"{c_final:.2f}")
        st.caption(
            f"Lambda={lambda_total:.2f}/h | TMA={tma_min:.2f} min | rho alvo={fila_diag.get('rho_target', rho_target):.2f} | "
            f"cálculo base={c_base:.2f} | cobertura(horas)={cobertura:.2f} | folga extra={folga_extra_pct:.1f}%"
        )
        if ativos:
            df_proc = pd.DataFrame(ativos)
            df_proc["freq_dia"] = df_proc["freq_semana"] / max(dias_oper, 1)
            st.dataframe(
                df_proc[
                    [
                        "Processo",
                        "tempo_min",
                        "freq_informada",
                        "freq_semana",
                        "freq_dia",
                        "lambda_hora",
                        "carga_horas_semana",
                    ]
                ],
                use_container_width=True,
            )
