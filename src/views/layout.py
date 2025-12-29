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
        @media (max-width: 1200px) {
            .main .block-container,
            [data-testid="block-container"] {
                padding-left: 64px !important;
                padding-right: 64px !important;
            }
        }
        @media (max-width: 900px) {
            .main .block-container,
            [data-testid="block-container"] {
                padding-left: 32px !important;
                padding-right: 32px !important;
            }
        }
        @media (max-width: 640px) {
            .main .block-container,
            [data-testid="block-container"] {
                padding-left: 16px !important;
                padding-right: 16px !important;
            }
        }
        /* Ajusta barras laterais (sidebar) */
        [data-testid="stSidebar"] {
            width: 240px;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        @media (max-width: 900px) {
            [data-testid="stSidebar"] {
                width: 200px;
            }
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
        /* Centraliza as abas principais da navbar */
        .stTabs [data-baseweb="tab-list"],
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"],
        .stTabs [role="tab"] {
            flex: 1 1 0;           /* mesmas larguras para as 3 abas */
            text-align: center;
        }
        /* Header responsivo */
        .kalk-header {
            width: 100%;
        }
        .kalk-header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            text-align: center;
        }
        .kalk-header-top img {
            max-width: 100%;
            height: auto;
        }
        .kalk-logo-nex,
        .kalk-logo-gentil {
            width: clamp(150px, 22vw, 220px);
        }
        .kalk-header-main {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 24px;
            flex-wrap: wrap;
            text-align: center;
            width: 100%;
            margin-bottom: 20px;
        }
        .kalk-logo-kalk {
            width: clamp(120px, 20vw, 180px);
            height: auto;
        }
        .kalk-header-text {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            max-width: 860px;
        }
        .kalk-header-title {
            margin: 0;
            font-size: clamp(1.4rem, 2.6vw, 2rem);
        }
        .kalk-header-desc {
            margin: 0;
            padding: 0 2.5rem;
            line-height: 1.5;
        }
        @media (max-width: 900px) {
            .kalk-header-main {
                gap: 16px;
            }
            .kalk-header-desc {
                padding: 0 1.5rem;
            }
        }
        @media (max-width: 640px) {
            .kalk-header-top {
                justify-content: center;
            }
            .kalk-header-desc {
                padding: 0;
            }
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
    <div class="kalk-header">
        <div class="kalk-header-top">
            <img class="kalk-logo-nex" src="data:image/svg+xml;base64,{logo_nex}">
            <img class="kalk-logo-gentil" src="data:image/png;base64,{logo_gentil}">
        </div>
        <hr>
        <div class="kalk-header-main">
            <img class="kalk-logo-kalk" src="data:image/svg+xml;base64,{logo_kalk}">
            <div class="kalk-header-text">
                <h2 class="kalk-header-title">Calculadora de Dimensionamento de Time</h2>
                <p class="kalk-header-desc">O modelo considera uma base consolidada de dados da operaÇõÇœo (caracterÇðsticas estruturais, fluxo de pedidos, desempenho comercial e padrÇæes de processos) para calcular tanto o <b>dimensionamento esperados pelo padrÇœo atual</b> quanto a <b>quantidade ideal estimada</b> de auxiliares.</p>
            </div>
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
            - **Qtd Aux Histórico** → mostra o dimensionamento esperado pelos padrões históricos e estruturais atuais da Gentil Negócios.
            - **Qtd Aux Ideal** → estima o dimensionamento ótimo com base em regressão estatística e comparação com lojas de perfil semelhante.

        - **3️⃣ Ajuste parâmetros complementares:**
            - *Margem de folga (%)* → compensação para variações operacionais.
            - *Fator de monotonia* → pondera rotinas repetitivas.
            - *Horas disponíveis por colaborador* → define o tempo produtivo no período analisado.

        - **4️⃣ Clique em “Calcular dimensionamento”:**
             O sistema apresentará:
            - **Qtd Aux Histórico ou Qtd Aux Ideal**, de acordo com o modelo escolhido e dados indicados;
            - e **indicadores de precisão** do modelo (R², MAPE, SMAPE e intervalo de confiança).
        """
        )
