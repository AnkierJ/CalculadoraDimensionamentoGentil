from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.logic import (
    image_to_base64,
    safe_float,
    _norm_code,
    get_lookup,
    get_lookup_value,
    _standardize_cols,
    _standardize_row,
    calc_pct,
    get_schema_dAmostras,
    get_schema_dEstrutura,
    get_schema_dPessoas,
    get_schema_fFaturamento,
    get_schema_fIndicadores,
    template_df,
    to_csv_bytes,
    read_csv_with_schema,
    validate_df,
    _filter_df_by_loja,
    _get_loja_row,
    agregar_tempo_medio_por_processo,
    prepare_training_dataframe,
    clean_training_dataframe,
    predict_qtd_auxiliares,
    get_total_reference_values,
    estimate_cluster_indicators,
    horas_operacionais_por_colaborador,
    infer_horas_loja_e_disp,
    carga_total_horas_loja,
    calcular_qtd_aux_ideal,
    ideal_simplificado_por_fluxo,
    agregar_tempo_medio_por_processo,
    inferir_frequencia_por_processo,
    evaluate_model_cv,
    predict_with_uncertainty, 
    collinearity_report, 
    _load_with_version,
    render_append,
    _train_cached,
    FEATURE_COLUMNS,
    MODEL_ALGO_NAMES,
    MODEL_ALGO_ORDER
)


DATA_DIR = Path(r"C:\Users\ankier.lima\Gentil Negócios\File server-GN - Comercial\Comercial 360\01. Dimensionamento do Time de Venda Direta\03. Calculadora")
if "dAmostras" not in st.session_state:
    st.session_state["dAmostras"] = _load_with_version(f"{DATA_DIR}/dAmostras.csv", "dAmostras")
if "dEstrutura" not in st.session_state:
    st.session_state["dEstrutura"] = _load_with_version(f"{DATA_DIR}/dEstrutura.csv", "dEstrutura")
if "dPessoas" not in st.session_state:
    st.session_state["dPessoas"] = _load_with_version(f"{DATA_DIR}/dPessoas.csv", "dPessoas")
if "fFaturamento" not in st.session_state:
    st.session_state["fFaturamento"] = _load_with_version(f"{DATA_DIR}/fFaturamento.csv", "fFaturamento")
if "fIndicadores" not in st.session_state:
    st.session_state["fIndicadores"] = _load_with_version(f"{DATA_DIR}/fIndicadores.csv", "fIndicadores")

path_amostras = Path(DATA_DIR) / "dAmostras.csv"
path_estrutura = Path(DATA_DIR) / "dEstrutura.csv"
path_pessoas = Path(DATA_DIR) / "dPessoas.csv"
path_faturamento = Path(DATA_DIR) / "fFaturamento.csv"
path_indicadores = Path(DATA_DIR) / "fIndicadores.csv"

# Configuração para evitar problemas de firewall
st.set_page_config(
    page_title="KALK ⱽᴰ", 
    page_icon= "src/assets/iconKALK.svg", 
    layout="centered"
)


# Styles utilizados na página
st.markdown("""
    <style>
        /* Forçar o radio a ocupar 100% */
        .stRadio > div { 
            width: 100% !important;
        }
        .stRadio [role="radiogroup"] {
            justify-content: space-between;  /* espalha os itens */
            width: 100% !important;
            display: flex !important;
        }
    </style>
""", unsafe_allow_html=True)

### ==========================  
## 🌟 HEADER
### ==========================
logo_nex = image_to_base64("src/assets/logoNEX.svg")
logo_gentil = image_to_base64("src/assets/logoGentil.png")
logo_kalk = image_to_base64("src/assets/logoKALK.svg")
st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        text-align:center;
        width:100%;
    ">
        <img src="data:image/svg+xml;base64,{logo_nex}" width="220">
        <img src="data:image/png;base64,{logo_gentil}" width="220">
    </div>
    <hr>
    <div style="
        display:flex;
        justify-self: start;
        justify-content: center;
        align-items:center;
        text-align:center;
        width:100%;
    ">
        <img src="data:image/svg+xml;base64,{logo_kalk}" width="180" style="margin-right: 20px">
        <div style="display: flex; flex-direction: column">
            <h2>Calculadora de Dimensionamento de Time</h2>
            <p>O modelo considera uma base consolidada de dados da operação (características estruturais, fluxo de pedidos, desempenho comercial e padrões de processos) para calcular tanto o <b>dimensionamento esperados pelo padrão atual</b> quanto a <b>quantidade ideal estimada</b> de auxiliares.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

### ==========================  
## ℹ️ TUTORIAL
### ==========================
with st.expander("ℹ️ Como usar a KALK?"):
    st.markdown(
        """
        - **1️⃣ Dados-base já carregados:**
         A calculadora já contém informações atualizadas de estrutura física, quadro de pessoas, faturamento, indicadores comerciais e tempos médios de processos.
         O envio de planilhas é **opcional**, usado apenas para **atualizar dados** ou **testar cenários personalizados**.
            - `dAmostras`: tempos médios por processo.
            - `dEstrutura`: área, prateleiras, caixas e horários de operação.
            - `dPessoas`: quadro atual de auxiliares e líderes.
            - `fFaturamento`: dados de pedidos, itens e faturamento.
            - `fIndicadores`: métricas comerciais (%Ativos, taxas, faturamento/hora etc.).

        - **2️⃣ Escolha o modo de cálculo:**
            - **Qtd Aux Atual** → mostra o dimensionamento esperado pelos padrões históricos e estruturais atuais da Gentil Negócios.
            - **Qtd Aux Ideal** → estima o dimensionamento ótimo com base em regressão estatística e comparação com lojas de perfil semelhante.

        - **3️⃣ Ajuste parâmetros complementares:**
            - *Margem de folga (%)* → compensação para variações operacionais.
            - *Fator de monotonia* → pondera rotinas repetitivas.
            - *Horas disponíveis por colaborador* → define o tempo produtivo no período analisado.

        - **4️⃣ Clique em “Calcular dimensionamento”:**
             O sistema apresentará:
            - a **carga total estimada** (em horas);
            - **Qtd Aux Atual ou Qtd Aux Ideal**, de acordo com o modelo escolhido e dados indicados;
            - e **indicadores de precisão** do modelo (R², MAPE, SMAPE e intervalo de confiança).
        """
    )

st.session_state["fIndicadores"] = _standardize_cols(st.session_state["fIndicadores"])

total_refs = get_total_reference_values(st.session_state.get("fIndicadores"))
total_base_ref = total_refs.get("BaseTotal", total_refs.get("Base", 0.0))
total_receita_ref = total_refs.get("ReceitaTotalMes", 0.0)

tab_calc, tab_dados = st.tabs(["Cálculo","Dados de base"])

def render_tab(nome: str, schema_fn):
    schema = schema_fn()
    col_a, col_b = st.columns([2,1])
    with col_a:
        up = st.file_uploader(f"Upload CSV para {nome}", type=["csv"], key=f"up_{nome}")
    with col_b:
        tmpl = template_df(schema)
        st.download_button(
            label="Baixar template CSV",
            data=to_csv_bytes(tmpl),
            file_name=f"{nome}_template.csv",
            mime="text/csv",
            key=f"dl_{nome}"
        )
    if up is not None:
        df_up = read_csv_with_schema(up, schema)
        ok, errs = validate_df(df_up, schema)
        if ok:
            st.session_state[nome] = df_up
            if nome == "fIndicadores":
                st.session_state[nome] = _standardize_cols(st.session_state[nome])
            st.success(f"{nome} atualizado via CSV. {len(st.session_state[nome])} linhas")
        else:
            st.error("; ".join(errs))
    st.dataframe(st.session_state[nome].head(200))


### ==========================  
## ⚙️ Configurações
### ==========================
with tab_calc:
    st.subheader("Configurações")
    st.subheader("Modo de cálculo")
    # -------------------------------
    # Escolha do método de cálculo 
    # -------------------------------
    opcoes = [
        "Histórico (Machine Learning)",
        "Ideal (Machine Learning)",
        "Ideal (Simplificado)",
    ]
    # valor padrão
    if "modo_calc" not in st.session_state:
        st.session_state.modo_calc = opcoes[0]
    def set_modo(modo):
        st.session_state.modo_calc = modo
    cols = st.columns(3)
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
    modo_calc = st.session_state.modo_calc  # use isso no restante do código
    
    #Tipo utilizado no Machine Learning
    if modo_calc == "Histórico (Machine Learning)":
        ref_mode = "historico"
    elif modo_calc == "Ideal (Machine Learning)":
        ref_mode = "ideal"
    
    st.divider()

    # -------------------------------
    # Informar dados de base
    # -------------------------------
    if modo_calc:
        col1, col2, col3 = st.columns(3)
        with col1:
            horas_disp_input = st.number_input(
                "Carga Horária Semanal por Auxiliar",
                min_value=5.0,
                value=44.0,
                step=1.0,
                format="%.1f",
                help="Horas efetivas semanais que cada auxiliar consegue entregar (já considerando %disp). Ex: 44h para jornada padrão."
            )
            horas_disp = float(horas_disp_input)
            if horas_disp_input > 200:
                horas_disp = horas_disp_input / 4.33
                st.caption(f"Valor informado parece mensal. Convertido para {horas_disp:.1f} h/semana.")
            horas_loja_config_raw = safe_float(
                st.session_state.get(
                    "horas_operacionais_form",
                    st.session_state.get("horas_loja_config", 60.0)
                ),
                60.0
            )
            # Se o valor está em horas diárias (<= 24), converte para semanal (assume 7 dias)
            if horas_loja_config_raw <= 24:
                horas_loja_config = horas_loja_config_raw * 6.0
            else:
                horas_loja_config = horas_loja_config_raw
        with col2:
            margem = st.number_input(
                "Margem de folga (0-1)",
                min_value=0.0,
                value=0.15,
                step=0.05,
                format="%.2f",
                help="Reserva adicional além dos buffers para cobrir sazonalidades ou baixa eficiência."
            )
        with col3:
            fator_monotonia = st.number_input(
                "Fator de monotonia",
                min_value=1.0,
                value=1.10,
                step=0.05,
                format="%.2f",
                help="Multiplicador aplicado à carga para refletir queda de produtividade por tarefas repetitivas."
            )

        st.session_state["horas_disp_semanais"] = horas_disp
        st.session_state["horas_loja_config"] = horas_loja_config

     # Dados de entrada para o método de cálculo ideal (Machine Learning / Simplificado)
    if modo_calc != "Histórico (Machine Learning)":
        colI1, colI2, colI3 = st.columns(3)
        with colI1:
            ocupacao_alvo = st.number_input(
                "Ocupação alvo (0–1)",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
                format="%.2f",
                help="Percentual do tempo produtivo em que o time deve ficar ocupado. Valores altos reduzem folga."
            )
        with colI2:
            absenteismo = st.number_input(
                "Absenteísmo (0–1)",
                min_value=0.00,
                max_value=0.30,
                value=0.08,
                step=0.01,
                format="%.2f",
                help="Proporção média de faltas, férias e treinamentos. Acrescentada ao dimensionamento."
            )
        with colI3:
            sla_buffer = st.number_input(
                "Folga p/ picos/SLA (0–1)",
                min_value=0.00,
                max_value=0.30,
                value=0.05,
                step=0.01,
                format="%.2f",
                help="Reserva para absorver picos de demanda e manter os SLAs sem acionar horas extras."
            )
        

### ==========================
## 🔍 Pesquisa de loja 
### ==========================
    st.markdown("**Pesquisar loja existente (opcional)**")
    # -------------------------------
    # Escolha do tipo de pesquisa
    # -------------------------------
    col_lookup = st.columns([1, 1, 1])
    with col_lookup[0]:
        lookup_field = st.radio(
            "",
            ["BCPS", "SAP", "Loja"],
            horizontal=True,
            key="lookup_field",
            label_visibility="collapsed"
        )

    # Recupera DataFrame de indicadores
    df_ind = st.session_state.get("fIndicadores")
    df_estrutura = st.session_state.get("dEstrutura")
    # -------------------------------
    # Campo de entrada de código ou loja
    # -------------------------------
    if lookup_field in ("BCPS", "SAP"):
        with col_lookup[1]:
            lookup_code = st.text_input("", placeholder=(f"Código ({lookup_field})"), key="lookup_code",label_visibility="collapsed")
    elif lookup_field == "Loja":
        with col_lookup[1]:
            user_input = st.text_input("", placeholder="Nome da loja", key="lookup_loja_input", label_visibility="collapsed")
        # valor padrão usado abaixo
        lookup_code = st.session_state.get("lookup_loja_input", "")

    # -------------------------------
    # Ação do botão Pesquisar
    # -------------------------------
    with col_lookup[2]:
        lookup_submit = st.button("Pesquisar", use_container_width=True)
    if lookup_submit:
        if ((df_ind is None or df_ind.empty) and
            (df_estrutura is None or df_estrutura.empty)):
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
                    if colname in ("BCPS","SAP"):
                        proceed = False
                series_norm = df_ind[colname].map(_norm_code)
                if colname != "Loja":
                    mask = series_norm == code_norm
                else:
                    mask = series_norm.str.contains(code_norm, na=False)
                matches = df_ind.loc[mask]
            else:
                if colname in ("BCPS","SAP"):
                    st.warning("⚠️ Base de indicadores não disponível para pesquisa por BCPS/SAP.")
                    proceed = False

            if not proceed and (df_estrutura is None or df_estrutura.empty):
                st.session_state["lookup_found"] = False
                st.session_state["lookup_row"] = None
            else:
                indicator_row = matches.iloc[0].to_dict() if not matches.empty else {}
                estrutura_row: Dict[str, object] = {}
                estrutura_used = False
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

                combined: Dict[str, object] = {}
                if estrutura_row:
                    combined.update(estrutura_row)
                if indicator_row:
                    combined.update(indicator_row)

                if combined:
                    st.session_state["lookup_found"] = True
                    st.session_state["lookup_row"] = _standardize_row(combined)
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

    # -------------------------------                
    # Feedback persistente da última busca
    # -------------------------------
    if st.session_state.get("lookup_found") and st.session_state.get("lookup_row"):
        loja_nome = str(st.session_state["lookup_row"].get("Loja", "")).strip()
        st.info(f"Usando indicadores da loja: **{loja_nome}**")


### ==========================
## 📊 Forms e Dados da Loja
### ==========================
    lookup_row = st.session_state.get("lookup_row")
    has_lookup = isinstance(lookup_row, dict) and len(lookup_row) > 0

    with st.form("form_inputs"):
        st.subheader("Dados da loja")
        # -------------------------------   
        # Seção: Estrutura Física
        # -------------------------------
        st.markdown("**Estrutura Física**")
        estrutura_defaults: Dict[str, float] = {}
        estrutura_flags: Dict[str, bool] = {}
        if has_lookup:
            for key in ["Area Total","Qtd Caixas","HorasOperacionais"]:
                val = safe_float(get_lookup(lookup_row, key), 0.0)
                if not pd.isna(val) and val is not None and val != 0.0:
                    estrutura_defaults[key] = val
            for key, col in [("Escritorio","Escritorio"),("Copa","Copa"),("Espaco Evento","Espaco Evento"),("Espaco Evento","Esp Conv")]:
                val = get_lookup(lookup_row, col)
                if isinstance(val, str):
                    estrutura_flags[key] = val.strip().upper() in ("SIM","VERDADEIRO","TRUE","1")
                else:
                    estrutura_flags[key] = bool(val)
        colA, colB, colC = st.columns(3)
        with colA:
            area_total = st.number_input(
                "Area Total",
                min_value=0.0,
                step=1.0,
                value=estrutura_defaults.get("Area Total", 0.0),
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
            qtd_caixas = st.number_input(
                "Qtd Caixas",
                min_value=0,
                step=1,
                value=int(estrutura_defaults.get("Qtd Caixas", 0.0)),
            )
            horas_op_default = float(estrutura_defaults.get("HorasOperacionais", 0.0) or st.session_state.get("horas_loja_config", 0.0))
            # Se o valor do CSV parece diário (<= 24), converte para semanal
            if horas_op_default > 0 and horas_op_default <= 24:
                horas_op_default = horas_op_default * 6.0
            horas_operacionais_input = st.number_input(
                "Horas operacionais (h/semana)",
                min_value=7.0,
                max_value=168.0,
                step=1.0,
                value=float(horas_op_default),
                format="%.0f",
                help="Tempo total semanal de funcionamento da loja (ex: 60h para 10h/dia x 6 dias). Alimenta os cálculos ideais/ML."
            )
        
        st.divider()
        
        # -------------------------------
        # Seção: Indicadores (preenchidos automaticamente se loja foi encontrada)
        # -------------------------------
        st.markdown("**Indicadores**")
        lookup_row = st.session_state.get("lookup_row")
        loja_nome_alvo = ""
        if has_lookup:
            loja_nome_alvo = str(lookup_row.get("Loja", "")).strip()
        else:
            loja_nome_alvo = str(st.session_state.get("lookup_loja_input", "")).strip()

        if has_lookup:
            base_total_val      = get_lookup_value("BaseTotal", ["Base"])
            base_ativa_val      = get_lookup_value("BaseAtiva")
            a0_val              = get_lookup_value("A0")
            a1aa3_val           = get_lookup_value("A1aA3")
            churn_val           = get_lookup_value("Churn")
            receita_total_val   = get_lookup_value("ReceitaTotalMes")
            reais_por_ativo_val = get_lookup_value("ReaisPorAtivo")
            atividade_er_val    = get_lookup_value("AtividadeER")
            inicios_val         = get_lookup_value("Inicios")
            reinicios_val       = get_lookup_value("Reinicios")
            recuperados_val     = get_lookup_value("Recuperados")
            i4a_i6_val          = get_lookup_value("I4aI6")
        else:
            base_total_val = 0.0
            base_ativa_val = 0.0
            a0_val = 0.0
            a1aa3_val = 0.0
            churn_val = 0.0
            receita_total_val = 0.0
            reais_por_ativo_val = 0.0
            atividade_er_val = 0.0
            inicios_val = 0.0
            reinicios_val = 0.0
            recuperados_val = 0.0
            i4a_i6_val = 0.0

        colIndA, colIndB, colIndC = st.columns(3)
        with colIndA:
            base_total = st.number_input("Base Total", min_value=0.0, step=1.0, value=base_total_val)
            base_ativa = st.number_input("Base Ativa", min_value=0.0, step=1.0, value=base_ativa_val)
            a0 = st.number_input("A0", min_value=0.0, step=1.0, value=a0_val)
            a1aA3 = st.number_input("A1 a A3", min_value=0.0, step=1.0, value=a1aa3_val)
        with colIndB:
            receita_total = st.number_input("Receita Total (R$)", min_value=0.0, step=100.0, value=receita_total_val, format="%.2f")
            reais_por_ativo = st.number_input("Reais por Ativo (R$)", min_value=0.0, step=1.0, value=reais_por_ativo_val, format="%.2f")
            atividade_er = st.number_input("Atividade ER", min_value=0.0, max_value=100.0, step=0.001, value=atividade_er_val, format="%.3f")
            churn = st.number_input("Churn", min_value=0.0, max_value=100.0, step=0.001, value=churn_val, format="%.3f")
        with colIndC:
            inicios = st.number_input("Inícios", min_value=0.0, step=1.0, value=inicios_val)
            reinicios = st.number_input("Reinícios", min_value=0.0, step=1.0, value=reinicios_val)
            recuperados = st.number_input("Recuperados", min_value=0.0, step=1.0, value=recuperados_val)
            i4_a_i6 = st.number_input("I4 a I6", min_value=0.0, step=1.0, value=i4a_i6_val)

        base_total_den = total_base_ref if total_base_ref and total_base_ref > 0 else (base_total if base_total > 0 else None)
        receita_total_den = total_receita_ref if total_receita_ref and total_receita_ref > 0 else (receita_total if receita_total > 0 else None)
        pct_base_total = calc_pct(base_total, base_total_den)
        pct_faturamento = calc_pct(receita_total, receita_total_den)
        pct_ativos = calc_pct(base_ativa, base_total if base_total > 0 else None)
        taxa_inicios = calc_pct(inicios, base_total if base_total > 0 else base_ativa)
        taxa_reativacao = calc_pct(recuperados, i4_a_i6)
        taxa_reinicio = calc_pct(reinicios, base_total if base_total > 0 else None)
        a0aA3 = a0 + a1aA3
        cluster_targets = ["Pedidos/Hora","Pedidos/Dia","Itens/Pedido","Faturamento/Hora","%Retirada"]
        cluster_values = {target: 0.0 for target in cluster_targets}

        def _estimate_pedidos_por_hora(vals: Dict[str, float], horas_operacionais_semanais: float) -> float:
            """
            Estima pedidos/hora considerando que horas_operacionais é semanal.
            Retorna pedidos/hora (média semanal).
            """
            base = float(vals.get("Pedidos/Hora") or 0.0)
            if base <= 0 and horas_operacionais_semanais > 0:
                pedidos_dia = float(vals.get("Pedidos/Dia") or 0.0)
                if pedidos_dia > 0:
                    # Converte pedidos/dia para pedidos/semana, depois calcula pedidos/hora média semanal
                    pedidos_semana = pedidos_dia * 6.0
                    base = pedidos_semana / max(horas_operacionais_semanais, 1.0)
            if base <= 0:
                faturamento_hora = float(vals.get("Faturamento/Hora") or 0.0)
                reais_por_ativo_local = float(vals.get("ReaisPorAtivo") or 0.0)
                if faturamento_hora > 0 and reais_por_ativo_local > 0:
                    base = faturamento_hora / max(reais_por_ativo_local, 1.0)
            return max(base, 0.0)
        cluster_result = None
        cluster_used = False

        if has_lookup:
            for target in cluster_targets:
                cluster_values[target] = safe_float(get_lookup(lookup_row, target), 0.0)
        else:
            manual_inputs = [base_total, base_ativa, receita_total, inicios, reinicios, recuperados, i4_a_i6]
            manual_has_data = any(val is not None and not pd.isna(val) and val > 0 for val in manual_inputs)
            if manual_has_data:
                cluster_inputs = {
                    "BaseAtiva": base_ativa,
                    "ReceitaTotalMes": receita_total,
                    "ReaisPorAtivo": reais_por_ativo,
                    "AtividadeER": atividade_er,
                    "Inicios": inicios,
                    "Reinicios": reinicios,
                    "Recuperados": recuperados,
                    "Churn": churn,
                    "A0": a0,
                    "A1aA3": a1aA3,
                    "A0aA3": a0aA3,
                    "I4aI6": i4_a_i6,
                }
                cluster_result = estimate_cluster_indicators(
                    st.session_state.get("fIndicadores"),
                    cluster_inputs,
                )
                if cluster_result:
                    cluster_used = True
                    for target in cluster_targets:
                        val = cluster_result.get(target)
                        cluster_values[target] = float(val) if val is not None else 0.0
                else:
                    st.warning("Não foi possível estimar os indicadores operacionais por clusterização. Verifique se fIndicadores possui dados suficientes.")
            else:
                st.info("Preencha os indicadores essenciais para estimar os indicadores operacionais por clusterização.")

        with st.expander("Indicadores derivados (cálculo automático)"):
            colDer1, colDer2, colDer3 = st.columns(3)
            with colDer1:
                st.metric("% da Base Total", f"{pct_base_total:.2f}%")
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

        st.session_state["horas_operacionais_form"] = float(horas_operacionais_input)

        features_input = {
            # Estrutura física
            "Area Total": area_total,
            "Qtd Caixas": qtd_caixas,
            "HorasOperacionais": float(horas_operacionais_input),
            "Escritorio": int(escritorio),
            "Copa": int(copa),
            "Espaco Evento": int(espaco_evento),
            # Indicadores (nomes precisam bater com FEATURE_COLUMNS)
            "BaseAtiva": base_ativa,
            "ReaisPorAtivo": reais_por_ativo,
            "%Ativos": pct_ativos,
            "TaxaInicios": taxa_inicios,
            "TaxaReativacao": taxa_reativacao,
            "Pedidos/Hora": pedidos_hora,
            "Pedidos/Dia": pedidos_dia,
            "Itens/Pedido": itens_pedido,
            "Faturamento/Hora": faturamento_hora,
            "%Retirada": pct_retirada,
        }

        # -----------------------------
        # Dados Manuseáveis (modo Simplificado)
        # -----------------------------
        if modo_calc == "Ideal (Simplificado)":
            st.divider()
            st.markdown("### Dados manuseáveis (simulação)")

            st.caption(
                "Os valores abaixo vêm **sugeridos** do histórico/clusterização quando possível, "
                "mas servem apenas como ponto de partida. "
                "Altere livremente para simular contextos de demanda e operação."
            )

            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_pedidos_dia = st.number_input(
                    "Pedidos/Dia (simulação)",
                    min_value=0.0,
                    step=1.0,
                    value=cluster_values["Pedidos/Dia"] if (cluster_used or has_lookup) else 0.0,
                    format="%.0f",
                    key="sim_pedidos_dia",
                    help="Volume total processado em um dia típico da loja. Use dados reais ou o cenário a testar."
                )
                sim_itens_pedido = st.number_input(
                    "Itens por pedido (simulação)",
                    min_value=0.0,
                    step=0.1,
                    value=cluster_values["Itens/Pedido"] if (cluster_used or has_lookup) else 0.0,
                    format="%.1f",
                    key="sim_itens_pedido",
                    help="Quantidade média de itens manipulados a cada pedido. Afeta o esforço por atendimento."
                )
            with sim_col2:
                tmedio_min_atend = st.number_input(
                    "Tempo médio de atendimento (min)",
                    min_value=0.0,
                    step=0.5,
                    value=10.0,  # valor padrão; ajuste se quiser puxar algo da sua base
                    format="%.1f",
                    key="tmedio_min_atend",
                    help="Tempo efetivo gasto para liberar um pedido completo (da chegada à entrega)."
                )
                sim_pct_retirada = st.number_input(
                    "% Retirada (simulação)",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    value=cluster_values["%Retirada"] if (cluster_used or has_lookup) else 0.0,
                    format="%.1f",
                    key="sim_pct_retirada",
                    help="Vendas em caixa também são consideradas retirada e devem ser consideradas."
                )
            with sim_col3:
                sim_faturamento_hora = st.number_input(
                    "Faturamento/Hora (simulação)",
                    min_value=0.0,
                    step=0.1,
                    value=cluster_values["Faturamento/Hora"] if (cluster_used or has_lookup) else 0.0,
                    format="%.2f",
                    key="sim_faturamento_hora",
                    help="Opcional. Use quando quiser alinhar a demanda via faturamento médio/hora; se 0, o sistema tenta inferir."
                )

            # Guarda no session_state para usar fora do form
            sim_payload = {
                "pedidos_dia": sim_pedidos_dia,
                "faturamento_hora": sim_faturamento_hora,
                "itens_pedido": sim_itens_pedido,
                "tmedio_min_atend": tmedio_min_atend,
                "pct_retirada": sim_pct_retirada,
            }
            st.session_state["sim_inputs"] = sim_payload
            st.session_state["dados_manuseaveis"] = sim_payload.copy()

  
    ### ========================== 
    ## Calcular QtdAux
    ### ==========================
        col1, col2 = st.columns([2,1])  # ajuste os pesos para mover
        with col2:
            submitted = st.form_submit_button("Calcular Qtd Auxiliares", type="primary", use_container_width=True)
    if submitted:
        # Descobre o nome da loja-alvo (se houver lookup)
        loja_nome_alvo = None
        if st.session_state.get("lookup_found") and st.session_state.get("lookup_row"):
            loja_nome_alvo = str(st.session_state["lookup_row"].get("Loja", "")).strip() or None

        # Prepara base de treinamento (para modos com ML)
        train_df = prepare_training_dataframe(
            st.session_state["dEstrutura"],
            st.session_state["dPessoas"],
            st.session_state["fIndicadores"],
        )
        train_df = clean_training_dataframe(train_df)
        if train_df.empty:
            st.error("Sem dados válidos para treinar os modelos. Verifique dEstrutura/dPessoas/fIndicadores.")
        elif len(train_df) < 15:
            st.info(f"A base de treino possui apenas {len(train_df)} lojas. As métricas podem variar bastante.")

        model_bundle = None
        # Só treina modelo quando for modo de ML
        if modo_calc in ("Histórico (Machine Learning)", "Ideal (Machine Learning)"):
            model_bundle = _train_cached(
                train_df,
                ref_mode,        # "historico" ou "ideal"
                horas_disp,      # vem lá da seção Configurações
                margem,          # idem
            )

        ## -------------------------------
        # 1) MODO HISTÓRICO (MACHINE LEARNING)
        ## -------------------------------
        if modo_calc == "Histórico (Machine Learning)":
            models = (model_bundle or {}).get("models", {}) if model_bundle else {}
            model_errors = dict((model_bundle or {}).get("errors", {})) if model_bundle else {}

            if not models:
                err_msgs = []
                for key, msg in model_errors.items():
                    label = MODEL_ALGO_NAMES.get(key, key) if key != "_geral" else "Modelo"
                    err_msgs.append(f"{label}: {msg}")
                detalhe = "; ".join(err_msgs) if err_msgs else "Faça upload de dEstrutura, dPessoas e (opcional) fIndicadores."
                st.error(f"Não há modelos treinados. {detalhe}")
            else:
                resultados_modelos = []
                for key in MODEL_ALGO_ORDER:
                    if key not in models:
                        continue
                    modelo_atual = models[key]
                    try:
                        pred = predict_qtd_auxiliares(modelo_atual, features_input)
                    except Exception as exc:
                        model_errors[key] = f"Erro ao prever: {exc}"
                        continue

                    metrics = evaluate_model_cv(
                        train_df,
                        n_splits=5,
                        mode=ref_mode,
                        horas_disp=horas_disp,
                        margem=margem,
                        algo=key,
                    )

                    resultados_modelos.append(
                        {
                            "key": key,
                            "label": MODEL_ALGO_NAMES.get(key, key),
                            "pred": pred,
                            "metrics": metrics or {},
                        }
                    )

                if resultados_modelos:
                    st.success("Cálculo concluído!")
                    st.subheader("Qtd Auxiliares estimada por modelo")
                    cols = st.columns(len(resultados_modelos))
                    for col, res in zip(cols, resultados_modelos):
                        precisao = res["metrics"].get("Precisao_percent")
                        r2m = res["metrics"].get("R2_mean")
                        with col:
                            st.metric(res["label"], f"{res['pred']:.2f}")
                            if precisao is not None:
                                st.caption(f"Precisão (1 - MAPE): {precisao:.1f}%")
                            if r2m is not None:
                                st.caption(f"R² médio: {r2m:.2f}")
                else:
                    st.error("Não foi possível gerar previsões para os modelos treinados.")

                if model_errors:
                    itens = []
                    for key, msg in model_errors.items():
                        if key == "_geral":
                            itens.append(msg)
                        else:
                            itens.append(f"{MODEL_ALGO_NAMES.get(key, key)}: {msg}")
                    st.info("Modelos indisponíveis ou com erro: " + "; ".join(itens))

                ci_model_key = next((k for k in MODEL_ALGO_ORDER if k in models), None)
                if ci_model_key:
                    ci = predict_with_uncertainty(
                        train_df,
                        features_input,
                        n_boot=200,
                        q=(5, 95),
                        mode=ref_mode,
                        horas_disp=horas_disp,
                        margem=margem,
                        algo=ci_model_key,
                    )
                    if ci:
                        label = MODEL_ALGO_NAMES.get(ci_model_key, ci_model_key)
                        st.write(
                            f"Intervalo de confiança (90%) para {label}: "
                            f"**{ci['ci_low']:.2f} – {ci['ci_high']:.2f}** "
                            f"(média bootstrap: {ci['pred_mean']:.2f})"
                        )

        ## -------------------------------
        # 2) MODOS IDEAL (ML E SIMPLIFICADO)
        ## -------------------------------
        else:
            result_ideal = None
            row_horas: Dict[str, object] = {}
            estrutura_df = st.session_state.get("dEstrutura")
            pessoas_df = st.session_state.get("dPessoas")

            estrutura_row, estrutura_match = _get_loja_row(estrutura_df, loja_nome_alvo)
            pessoas_row, pessoas_match = _get_loja_row(pessoas_df, loja_nome_alvo)

            if estrutura_row:
                row_horas.update(estrutura_row)
            if pessoas_row and "%disp" in pessoas_row:
                row_horas["%disp"] = pessoas_row["%disp"]

            manual_horas_form = safe_float(st.session_state.get("horas_operacionais_form"), 0.0)
            if manual_horas_form > 0:
                # Se o valor informado parece diário (<= 24), converte para semanal
                if manual_horas_form <= 24:
                    manual_horas_form = manual_horas_form * 6.0
                row_horas["HorasOperacionais"] = manual_horas_form

            has_row_data = bool(row_horas)
            if loja_nome_alvo:
                use_loja_dados = bool(estrutura_match or pessoas_match)
            else:
                use_loja_dados = has_row_data

            row_series = pd.Series(row_horas) if row_horas else None
            horas_por_colab = float(st.session_state.get("horas_disp_semanais", horas_disp))
            horas_loja_manual = float(st.session_state.get("horas_loja_config", horas_loja_config))
            # Se horas_loja_manual parece diário (<= 24), converte para semanal
            if horas_loja_manual <= 24:
                horas_loja_manual = horas_loja_manual * 6.0
            horas_loja = max(5.0, min(168.0, horas_loja_manual))  # min 5h, max 168h (7*24)

            if use_loja_dados and row_series is not None and not row_series.empty:
                horas_calc = horas_operacionais_por_colaborador(row_series)
                if pd.notna(horas_calc) and horas_calc > 0:
                    # horas_calc vem diário (h/dia * %disp), precisa converter para semanal
                    horas_calc_semanal = float(horas_calc) * 6.0
                    horas_por_colab = horas_calc_semanal

                loja_hours, _ = infer_horas_loja_e_disp(row_series)
                if pd.notna(loja_hours) and loja_hours > 0:
                    # loja_hours vem diário, converte para semanal
                    horas_loja = float(loja_hours) * 6.0
            else:
                horas_raw = float(row_horas.get("HorasOperacionais", 0.0) or horas_loja_manual)
                # Se parece diário, converte para semanal
                if horas_raw <= 24:
                    horas_raw = horas_raw * 6.0
                horas_loja = horas_raw

            if horas_loja <= 0:
                horas_loja = horas_loja_manual
            horas_loja = max(5.0, min(168.0, horas_loja))

            # -------------------------------
            # 2.1) Ideal (Machine Learning)
            # -------------------------------
            if modo_calc == "Ideal (Machine Learning)":
                models = (model_bundle or {}).get("models", {}) if model_bundle else {}
                model_errors = dict((model_bundle or {}).get("errors", {})) if model_bundle else {}

                if not models:
                    err_msgs = []
                    for key, msg in model_errors.items():
                        label = MODEL_ALGO_NAMES.get(key, key) if key != "_geral" else "Modelo"
                        err_msgs.append(f"{label}: {msg}")
                    detalhe = "; ".join(err_msgs) if err_msgs else "Faça upload de dEstrutura, dPessoas e fIndicadores suficientes."
                    st.error(f"Não há modelos treinados (Ideal). {detalhe}")
                else:
                    resultados_modelos_ideal: List[Dict[str, object]] = []
                    for key in MODEL_ALGO_ORDER:
                        if key not in models:
                            continue
                        modelo_atual = models[key]
                        try:
                            pred = predict_qtd_auxiliares(modelo_atual, features_input)
                        except Exception as exc:
                            model_errors[key] = f"Erro ao prever: {exc}"
                            continue

                        metrics = evaluate_model_cv(
                            train_df,
                            n_splits=5,
                            mode=ref_mode,
                            horas_disp=horas_disp,
                            margem=margem,
                            algo=key,
                        )

                        resultados_modelos_ideal.append(
                            {
                                "key": key,
                                "label": MODEL_ALGO_NAMES.get(key, key),
                                "pred": pred,
                                "metrics": metrics or {},
                            }
                        )

                    if resultados_modelos_ideal:
                        st.success("Previsão (Ideal ML) concluída!")
                        st.subheader("Qtd Auxiliares (ideal) por modelo")
                        cols = st.columns(len(resultados_modelos_ideal))
                        for col, res in zip(cols, resultados_modelos_ideal):
                            precisao = res["metrics"].get("Precisao_percent")
                            r2m = res["metrics"].get("R2_mean")
                            with col:
                                st.metric(res["label"], f"{res['pred']:.2f}")
                                if precisao is not None:
                                    st.caption(f"Precisão (1 - MAPE): {precisao:.1f}%")
                                if r2m is not None:
                                    st.caption(f"R² médio: {r2m:.2f}")
                    else:
                        st.error("Não foi possível gerar previsões para os modelos ideais treinados.")

                    ci_model_key = next((k for k in MODEL_ALGO_ORDER if k in models), None)
                    if ci_model_key:
                        ci = predict_with_uncertainty(
                            train_df,
                            features_input,
                            n_boot=200,
                            q=(5, 95),
                            mode=ref_mode,
                            horas_disp=horas_disp,
                            margem=margem,
                            algo=ci_model_key,
                        )
                        if ci:
                            label = MODEL_ALGO_NAMES.get(ci_model_key, ci_model_key)
                            st.write(
                                f"Intervalo de confiança (90%) para {label}: "
                                f"**{ci['ci_low']:.2f} – {ci['ci_high']:.2f}** "
                                f"(média bootstrap: {ci['pred_mean']:.2f})"
                            )

                amostras_df = st.session_state.get("dAmostras")
                amostras_loja, _ = _filter_df_by_loja(amostras_df, loja_nome_alvo)

                tempos = agregar_tempo_medio_por_processo(amostras_loja)
                freq_df = inferir_frequencia_por_processo(amostras_loja)

                detalhe, carga_total_diaria = carga_total_horas_loja(
                    tempos_processo=tempos,
                    frequencias=freq_df,
                    fator_monotonia=fator_monotonia
                )
                # Converte carga diária para semanal (multiplica por 7 dias)
                carga_total = carga_total_diaria * 6.0

                if carga_total <= 0:
                    st.info("Não foi possível estimar a carga pelos processos; usando fluxo simplificado.")
                    pedidos_hora_flow = _estimate_pedidos_por_hora(cluster_values, horas_loja)
                    _ = ideal_simplificado_por_fluxo(  # cálculo apenas de apoio
                        pedidos_hora=pedidos_hora_flow,
                        horas_operacionais_loja=horas_loja,
                        tmedio_min_atendimento=tmedio_min_atend,
                        fator_monotonia=fator_monotonia,
                        horas_por_colaborador=horas_por_colab,
                        margem_operacional=margem,
                        ocupacao_alvo=ocupacao_alvo,
                        absenteismo=absenteismo,
                        sla_buffer=sla_buffer
                    )
                else:
                    _ = calcular_qtd_aux_ideal(
                        carga_total_horas=carga_total,
                        horas_por_colaborador=horas_por_colab,
                        margem_operacional=margem,
                        ocupacao_alvo=ocupacao_alvo,
                        absenteismo=absenteismo,
                        sla_buffer=sla_buffer
                    )

            # -------------------------------
            # 2.2) Ideal (Simplificado)  → "modelo simulado"
            # -------------------------------
            elif modo_calc == "Ideal (Simplificado)":
                # aqui você pode usar diretamente os dados manuseáveis
                # ou continuar usando os clusterizados como default
                pedidos_hora_sim = safe_float(cluster_values["Pedidos/Hora"], 0.0)
                sim_inputs = st.session_state.get("sim_inputs", {})
                sim_pedidos_dia = safe_float(sim_inputs.get("pedidos_dia"), 0.0)
                sim_tmedio = safe_float(sim_inputs.get("tmedio_min_atend"), tmedio_min_atend)
                sim_faturamento_hora = safe_float(sim_inputs.get("faturamento_hora"), 0.0)

                ticket_medio_ref = 0.0
                cluster_pedidos_hora = safe_float(cluster_values.get("Pedidos/Hora"), 0.0)
                cluster_faturamento_hora = safe_float(cluster_values.get("Faturamento/Hora"), 0.0)
                cluster_pedidos_dia = safe_float(cluster_values.get("Pedidos/Dia"), 0.0)
                if cluster_pedidos_hora > 0 and cluster_faturamento_hora > 0:
                    ticket_medio_ref = cluster_faturamento_hora / max(cluster_pedidos_hora, 1e-6)
                elif cluster_pedidos_dia > 0 and cluster_faturamento_hora > 0 and horas_loja > 0:
                    # horas_loja é semanal, mas faturamento_hora é por hora, precisa converter
                    # Assumindo que faturamento/hora * horas semanais = faturamento semanal
                    faturamento_semana = cluster_faturamento_hora * max(horas_loja, 1.0)
                    pedidos_semana = cluster_pedidos_dia * 6.0
                    ticket_medio_ref = faturamento_semana / max(pedidos_semana, 1e-6)

                pedidos_hora_manual = 0.0
                if sim_pedidos_dia > 0 and horas_loja > 0:
                    # horas_loja é semanal, precisa converter pedidos_dia para semanal
                    pedidos_semana = sim_pedidos_dia * 6.0
                    pedidos_hora_manual = pedidos_semana / max(horas_loja, 1.0)
                elif sim_faturamento_hora > 0 and ticket_medio_ref > 0:
                    pedidos_hora_manual = sim_faturamento_hora / ticket_medio_ref

                pedidos_hora_sim_aj = _estimate_pedidos_por_hora(cluster_values, horas_loja)
                pedidos_hora_final = pedidos_hora_manual if pedidos_hora_manual > 0 else pedidos_hora_sim
                if pedidos_hora_final <= 0:
                    pedidos_hora_final = pedidos_hora_sim_aj

                tmedio_utilizado = sim_tmedio if sim_tmedio > 0 else tmedio_min_atend
                result_ideal = ideal_simplificado_por_fluxo(
                    pedidos_hora=pedidos_hora_final,
                    horas_operacionais_loja=horas_loja,
                    tmedio_min_atendimento=tmedio_utilizado,
                    fator_monotonia=fator_monotonia,
                    horas_por_colaborador=horas_por_colab,
                    margem_operacional=margem,
                    ocupacao_alvo=ocupacao_alvo,
                    absenteismo=absenteismo,
                    sla_buffer=sla_buffer
                )

            else:
                # fallback de segurança, caso apareça algum modo inesperado
                st.error(f"Modo de cálculo não reconhecido: {modo_calc}")
                result_ideal = None

            if modo_calc == "Ideal (Simplificado)" and result_ideal is not None:
                st.success("Cálculo (Ideal) concluído!")
                st.metric("Qtd Auxiliares (ideal)", f"{result_ideal['qtd_aux_ideal']}")
                st.caption(
                    f"Carga: {result_ideal['carga_total_horas']:.2f} h/semana | "
                    f"H/colab: {result_ideal['horas_por_colaborador']:.2f} h/semana | "
                    f"Ocupação alvo: {result_ideal['ocupacao_alvo']:.2f} | "
                    f"Absenteísmo: {result_ideal['absenteismo']:.2f} | "
                    f"SLA buffer: {result_ideal['sla_buffer']:.2f} | "
                    f"Margem: {result_ideal['margem']:.2f}"
                )
                st.caption(
                    f"Pedidos/h usados: {result_ideal.get('pedidos_hora_utilizado', 0.0):.2f} | "
                    f"Tempo médio: {result_ideal.get('tmedio_min_atendimento', 0.0):.2f} min | "
                    f"Fator monotonia: {result_ideal.get('fator_monotonia', fator_monotonia):.2f}"
                )


### ==========================
## 🩺 Diagóstico
### ==========================
    st.subheader("Diagnóstico (opcional)")
    if st.button("Checar colinearidade das features"):
        train_df_diag = prepare_training_dataframe(
            st.session_state["dEstrutura"],
            st.session_state["dPessoas"],
            st.session_state["fIndicadores"],
        )
        train_df_diag = clean_training_dataframe(train_df_diag)
        from src.logic import collinearity_report, FEATURE_COLUMNS
        if train_df_diag.empty:
            st.warning("Sem dados suficientes para avaliar colinearidade.")
        else:
            feat_disp = [c for c in FEATURE_COLUMNS if c in train_df_diag.columns]
            st.caption(f"{len(train_df_diag)} lojas válidas | {len(feat_disp)} variáveis analisadas.")
            rep = collinearity_report(train_df_diag, FEATURE_COLUMNS, corr_thr=0.85)
            st.write("🔁 Pares altamente correlacionados (|r|≥0.85):")
            if rep["high_corr_pairs"]:
                st.dataframe(pd.DataFrame(rep["high_corr_pairs"], columns=["col1","col2","r"]), use_container_width=True)
            else:
                st.success("Nenhum par acima do limiar.")
            st.write("📐 VIF por coluna (ideal < 10):")
            st.dataframe(pd.DataFrame(rep["vif"], columns=["col","VIF"]), use_container_width=True)
### ==========================
## 🔄 Limpar Cache
### ==========================
    st.subheader("Re-treinar modelo (opcional)")
    if st.button("🔄 Re-treinar (limpar cache)"):
        _train_cached.clear()
    st.success("Cache limpo. O próximo cálculo irá re-treinar o modelo.")



### ==========================
## 🗃️ Aba de Dados
### ==========================
with tab_dados:
    st.subheader("Dados de base")
    aba_preview, aba_upload = st.tabs(["Base atual (somente leitura)", "Adicionar dados (upload)"])
    with aba_preview:
        st.caption("Pré-visualização das 10 primeiras linhas das tabelas carregadas por padrão do diretório 'data/'.")
        with st.expander("dAmostras"):
            st.dataframe(st.session_state["dAmostras"].head(10), use_container_width=True)
            if path_amostras.exists():
                st.write(f"📦 Tamanho de dAmostras.csv: {path_amostras.stat().st_size:,} bytes")
            else:
                st.warning("Arquivo dAmostras.csv não encontrado.")
        with st.expander("dEstrutura"):
            st.dataframe(st.session_state["dEstrutura"].head(10), use_container_width=True)
            if path_estrutura.exists():
                st.write(f"📦 Tamanho de dEstrutura.csv: {path_estrutura.stat().st_size:,} bytes")
            else:
                st.warning("Arquivo dEstrutura.csv não encontrado.")
        with st.expander("dPessoas"):
            st.dataframe(st.session_state["dPessoas"].head(10), use_container_width=True)
            if path_pessoas.exists():
                st.write(f"📦 Tamanho de dPessoas.csv: {path_pessoas.stat().st_size:,} bytes")
            else:
                st.warning("Arquivo dPessoas.csv não encontrado.")
        with st.expander("fFaturamento"):
            st.dataframe(st.session_state["fFaturamento"].head(10), use_container_width=True)
            if path_faturamento.exists():
                st.write(f"📦 Tamanho de fFaturamento.csv: {path_faturamento.stat().st_size:,} bytes")
            else:
                st.warning("Arquivo fFaturamento.csv não encontrado.")
        with st.expander("fIndicadores"):
            st.dataframe(st.session_state["fIndicadores"].head(10), use_container_width=True)
            if path_indicadores.exists():
                st.write(f"📦 Tamanho de fIndicadores.csv: {path_indicadores.stat().st_size:,} bytes")
            else:
                st.warning("Arquivo fIndicadores.csv não encontrado.")
        st.divider()
        colx1, colx2, colx3 = st.columns([1,3,1], gap="medium", vertical_alignment="center")
        with colx2:
            if st.button("Gerar arquivos para download", use_container_width=True, type="primary"):
                st.session_state["downloads_ready"] = True

        if st.session_state.get("downloads_ready"):
            for nome in ["dAmostras","dEstrutura","dPessoas","fFaturamento","fIndicadores"]:
                df = st.session_state[nome]
                csv_bytes = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                st.download_button(
                    label=f"⬇️ Baixar {nome}.csv",
                    data=csv_bytes,
                    file_name=f"{nome}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with aba_upload:
        st.caption("Envie arquivos para ACRESCENTAR dados à base atual. Entradas duplicadas são deduplicadas por chaves básicas.")
        tabs = st.tabs(["dAmostras", "dEstrutura", "dPessoas", "fFaturamento", "fIndicadores"])
        with tabs[0]:
            render_append("dAmostras", get_schema_dAmostras, ["Loja", "Processo", "Amostra"]) 
        with tabs[1]:
            render_append("dEstrutura", get_schema_dEstrutura, ["Loja"]) 
        with tabs[2]:
            render_append("dPessoas", get_schema_dPessoas, ["Loja"]) 
        with tabs[3]:
            render_append("fFaturamento", get_schema_fFaturamento, ["Loja", "CodPedido"]) 
        with tabs[4]:
            render_append("fIndicadores", get_schema_fIndicadores, ["Loja"]) 
