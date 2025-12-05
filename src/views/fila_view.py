import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.data.buscaDeLojas import _filter_df_by_loja
from src.logic.core.logic import (
    DEFAULT_ABSENTEISMO,
    WEEKS_PER_MONTH,
    agregar_tempo_medio_por_processo,
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

        tempos_df_all = agregar_tempo_medio_por_processo(amostras_df)
        freq_df_all = pd.DataFrame(columns=["Processo", "frequencia"])
        if tempos_df_all.empty:
            st.warning("Sem tempos médios calculados em dAmostras.")
            return

        tempos_df_loja = tempos_df_all
        freq_df_loja = freq_df_all
        if loja_escolhida != "Média geral":
            amostras_loja, _ = _filter_df_by_loja(amostras_df, loja_escolhida)
            if not amostras_loja.empty:
                tempos_df_loja = agregar_tempo_medio_por_processo(amostras_loja)
                freq_df_loja = pd.DataFrame(columns=["Processo", "frequencia"])
            else:
                st.warning("Não encontrei amostras para essa loja; usando média geral.")

        def _norm_proc_nome(nome: str) -> str:
            base = str(nome or "").strip().lower()
            base = base.split("(")[0].strip()
            return " ".join(base.split())

        def _prep_cols(df: pd.DataFrame | None, col: str, tag: str, keep_process: bool = True) -> pd.DataFrame:
            if df is None or df.empty:
                cols = ["proc_norm", f"{col}_{tag}"]
                if keep_process:
                    cols.append(f"Processo_{tag}")
                return pd.DataFrame(columns=cols)
            tmp = df[["Processo", col]].copy()
            tmp["proc_norm"] = tmp["Processo"].apply(_norm_proc_nome)
            tmp = (
                tmp.groupby("proc_norm", dropna=False)
                .agg({col: "mean", "Processo": lambda s: sorted(s, key=len)[0]})
                .reset_index()
            )
            rename_map = {col: f"{col}_{tag}"}
            if keep_process:
                rename_map["Processo"] = f"Processo_{tag}"
            else:
                tmp = tmp.drop(columns=["Processo"])
            return tmp.rename(columns=rename_map)

        tempo_loja = _prep_cols(tempos_df_loja, "tempo_medio_min", "loja")
        tempo_geral = _prep_cols(tempos_df_all, "tempo_medio_min", "all")
        freq_loja = _prep_cols(freq_df_loja, "frequencia", "loja", keep_process=False)
        freq_geral = _prep_cols(freq_df_all, "frequencia", "all", keep_process=False)

        processos_df = tempo_geral.merge(tempo_loja, on="proc_norm", how="outer")
        processos_df = processos_df.merge(freq_geral, on="proc_norm", how="outer")
        processos_df = processos_df.merge(freq_loja, on="proc_norm", how="outer")
        if processos_df.empty:
            st.warning("Nenhum processo encontrado em dAmostras.")
            return

        for col in ["tempo_medio_min_loja", "tempo_medio_min_all", "frequencia_loja", "frequencia_all"]:
            processos_df[col] = pd.to_numeric(processos_df[col], errors="coerce")

        processos_df["Processo"] = (
            processos_df["Processo_loja"]
            .combine_first(processos_df["Processo_all"])
            .fillna("Processo sem nome")
        )
        processos_df["tempo_medio_min"] = (
            processos_df["tempo_medio_min_loja"]
            .combine_first(processos_df["tempo_medio_min_all"])
            .fillna(0.0)
        )
        processos_df["frequencia"] = (
            processos_df["frequencia_loja"]
            .combine_first(processos_df["frequencia_all"])
            .fillna(0.0)
        )
        processos_df["usa_media_geral"] = (
            (processos_df["tempo_medio_min_loja"].isna() & processos_df["tempo_medio_min_all"].notna())
            | (processos_df["frequencia_loja"].isna() & processos_df["frequencia_all"].notna())
        )

        ordem_processos = [
            "Devolução",
            "Reposição de prateleira",
            "Produção de flyer",
            "Abertura e acompanhamento de chamado",
            "ação de VPs/Excesso",
            "Criação de conteúdo",
            "elaboração de calendário do ciclo e divulgação",
            "Encontro de ciclo",
            "Eventos para os revendedores",
            "Unibê",
            "Limpeza da ER",
            "Limpeza das salas e Copa",
            "Limpeza dos banheiros",
            "Mudança de planograma",
            "Fechamento de caixa",
            "Mudança de vitrine",
            "Atualização de cadastro de revendedor",
            "Cadastro de revendedor",
            "digitalização de boletos",
            "Faturamente de pedido (retirada e delivery)",
            "Separação de mercadoria (on-line e retirada)",
            "Venda em caixa",
            "Atendimento ao cliente",
            "Prospecção de revendedor (Início)",
            "Atendimento online",
            "Ativações on-line",
        ]
        ordem_map = {_norm_proc_nome(p): i for i, p in enumerate(ordem_processos)}
        processos_df["proc_norm"] = processos_df["Processo"].apply(_norm_proc_nome)
        processos_df = (
            processos_df.groupby("proc_norm", dropna=False)
            .agg(
                {
                    "Processo": lambda s: sorted(s, key=len)[0] if not s.empty else "Processo",
                    "tempo_medio_min": "mean",
                    "frequencia": "mean",
                    "usa_media_geral": "any",
                }
            )
            .reset_index()
        )
        processos_df["ordem_custom"] = processos_df["proc_norm"].apply(
            lambda x: ordem_map.get(x, len(ordem_map))
        )
        processos_df = processos_df.sort_values(["ordem_custom", "Processo"])

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
                    "Horas contratuais (h/sem)",
                    min_value=1.0,
                    step=1.0,
                    value=44.0,
                    format="%.1f",
                    help="Usado para ajustar headcount quando a carga semanal da loja excede essa jornada.",
                )
                absenteismo = st.number_input(
                    "Absenteísmo (0–1)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    value=float(DEFAULT_ABSENTEISMO),
                    format="%.2f",
                    help="Percentual de horas perdidas por faltas/férias/treinamentos. Será abatido das horas contratuais.",
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
            for _, proc_row in processos_df.iterrows():
                proc_nome = str(proc_row["Processo"]).strip() or "Processo"
                proc_norm = proc_nome.lower().strip()
                defaults = state_map.get(proc_norm, {})
                with st.container():
                    fallback_label = " _(média geral)_" if proc_row.get("usa_media_geral") else ""
                    st.markdown(f"**{proc_nome}**{fallback_label}")
                    col_proc1, col_proc2 = st.columns(2)
                    with col_proc1:
                        tempo_val = st.number_input(
                            "Tempo médio (min)",
                            min_value=0.0,
                            value=float(defaults.get("tempo", proc_row["tempo_medio_min"] or 0.0)),
                            step=0.5,
                            format="%.2f",
                            key=f"fila_tempo_{state_key}_{proc_norm}",
                        )
                    with col_proc2:
                        freq_val = st.number_input(
                            freq_label,
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
                carga_semana_min = freq_semana * tempo_val
                carga_semana_horas = carga_semana_min / 60.0
                processos_resultados.append(
                    {
                        "Processo": proc_nome,
                        "tempo_min": tempo_val,
                        "freq_informada": freq_val,
                        "freq_semana": freq_semana,
                        "lambda_hora": lambda_proc,
                        "carga_semana_min": carga_semana_min,
                        "carga_semana_horas": carga_semana_horas,
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
        total_carga_semana_horas = sum(p["carga_semana_horas"] for p in ativos)
        abs_ratio = max(1.0 - absenteismo, 1e-3)
        horas_produtivas = horas_contratuais * abs_ratio
        cobertura = horas_semana_oper / horas_produtivas if horas_semana_oper > 0 else 1.0
        cobertura = max(cobertura, 1.0)
        c_base = float(fila_diag.get("c_fila", 0.0) or 0.0)
        c_ajust = c_base * cobertura
        c_final = c_ajust * (1 + folga_extra_pct / 100)

        st.success("Resultado da fila")
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.metric("Lambda total (chegadas/h)", f"{lambda_total:.2f}")
        with col_res2:
            st.metric("TMA ponderado (min)", f"{tma_min:.2f}")
        with col_res3:
            st.metric("Headcount bruto (fila)", f"{c_base:.2f}")
        with col_res4:
            st.metric("Carga semanal total (h)", f"{total_carga_semana_horas:.2f}")
        st.metric("Headcount ajustado (absenteísmo/horas)", f"{c_final:.2f}")
        st.caption(
            f"Lambda={lambda_total:.2f}/h | TMA={tma_min:.2f} min | rho alvo={fila_diag.get('rho_target', rho_target):.2f} | "
            f"cálculo base={c_base:.2f} | carga/semana total={total_carga_semana_horas:.2f} h | absenteísmo aplicado={absenteismo*100:.1f}% | cobertura(horas)={cobertura:.2f} | folga extra={folga_extra_pct:.1f}%"
        )
        if ativos:
            df_proc = pd.DataFrame(ativos)
            df_proc["freq_dia"] = df_proc["freq_semana"] / max(dias_oper, 1)
            df_proc["carga_semana_horas"] = df_proc["carga_semana_min"] / 60.0
            freq_informada_label = f"Freq. informada ({freq_label})"
            df_proc = df_proc.rename(
                columns={
                    "tempo_min": "Tempo (min)",
                    "freq_informada": freq_informada_label,
                    "freq_semana": "Freq. semana",
                    "freq_dia": "Freq. dia",
                    "lambda_hora": "Lambda (h)",
                    "carga_semana_min": "Carga/semana (min)",
                    "carga_semana_horas": "Carga/semana (h)",
                }
            )
            st.dataframe(
                df_proc[
                    [
                        "Processo",
                        "Tempo (min)",
                        freq_informada_label,
                        "Freq. semana",
                        "Freq. dia",
                        "Lambda (h)",
                        "Carga/semana (min)",
                        "Carga/semana (h)",
                    ]
                ],
                use_container_width=True,
            )
