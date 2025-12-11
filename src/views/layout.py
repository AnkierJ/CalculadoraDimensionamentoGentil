import streamlit as st

from src.logic.utils.helpers import image_to_base64


def set_page_config() -> None:
    """Configurações globais da página."""
    st.set_page_config(
        page_title="KALK ⱽᴰ",
        page_icon="src/assets/iconKALK.svg",
        layout="wide",  # ocupar toda a largura da viewport
    )


def inject_global_styles() -> None:
    """Injeta estilos globais usados no app."""
    st.markdown(
        """
    <style>
        /* Reduz padding e remove limite de largura do container principal */
        .main .block-container,
        [data-testid="block-container"] {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 120px !important;
            padding-right: 120px !important;
        }
        /* Ajusta barras laterais (sidebar) */
        [data-testid="stSidebar"] {
            width: 240px;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
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
    """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Renderiza o cabeçalho com logos."""
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
        unsafe_allow_html=True,
    )


def render_tutorial() -> None:
    """Renderiza o tutorial em formato de expander."""
    with st.expander("ℹ️ Como usar a KALK?"):
        st.markdown(
            """
        - **1️⃣ Dados-base já carregados:**
         A calculadora já contém informações atualizadas de estrutura física, quadro de pessoas, faturamento, indicadores comerciais e tempos médios de processos.
         O envio de planilhas é **opcional**, usado apenas para **atualizar dados** ou **testar cenários personalizados**.
            - `dAmostras`: tempos médios por processo.
            - `dEstrutura`: área, prateleiras, caixas e horários de operação.
            - `dPessoas`: quadro atual de auxiliares e líderes.
            - `fFaturamento2`: dados de pedidos, itens e faturamento.
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
