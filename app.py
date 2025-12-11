import streamlit as st
from pathlib import Path

from src.logic.core.logic import _load_with_version
from src.logic.utils.helpers import _standardize_cols
from src.views.calc_view import render_calc_tab
from src.views.comparativo_view import render_comparativo_tab
from src.views.dados_view import render_dados_tab
from src.views.diagnostics_view import render_diag_cache
from src.views.fila_view import render_fila_tab
from src.views.layout import inject_global_styles, render_header, render_tutorial, set_page_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Carregamento inicial dos datasets em session_state
if "dAmostras" not in st.session_state:
    st.session_state["dAmostras"] = _load_with_version(f"{DATA_DIR}/dAmostras.csv", "dAmostras")
if "dEstrutura" not in st.session_state:
    st.session_state["dEstrutura"] = _load_with_version(f"{DATA_DIR}/dEstrutura.csv", "dEstrutura")
if "dPessoas" not in st.session_state:
    st.session_state["dPessoas"] = _load_with_version(f"{DATA_DIR}/dPessoas.csv", "dPessoas")
if "fFaturamento2" not in st.session_state:
    st.session_state["fFaturamento2"] = _load_with_version(f"{DATA_DIR}/fFaturamento2.csv", "fFaturamento2")
if "fIndicadores" not in st.session_state:
    st.session_state["fIndicadores"] = _load_with_version(f"{DATA_DIR}/fIndicadores.csv", "fIndicadores")

# Paths e dicionário para uso nas views
path_amostras = DATA_DIR / "dAmostras.csv"
path_estrutura = DATA_DIR / "dEstrutura.csv"
path_pessoas = DATA_DIR / "dPessoas.csv"
path_faturamento = DATA_DIR / "fFaturamento2.csv"
path_indicadores = DATA_DIR / "fIndicadores.csv"
paths = {
    "dAmostras": path_amostras,
    "dEstrutura": path_estrutura,
    "dPessoas": path_pessoas,
    "fFaturamento2": path_faturamento,
    "fIndicadores": path_indicadores,
}

set_page_config()
inject_global_styles()
render_header()
render_tutorial()

# Normalização das colunas de indicadores
st.session_state["fIndicadores"] = _standardize_cols(st.session_state["fIndicadores"])

# Tabs principais
tab_calc, tab_comp, tab_fila, tab_dados = st.tabs(["Cálculo", "Comparativo", "Teoria das filas", "Dados de base"])

with tab_calc:
    render_calc_tab(tab_calc)

with tab_comp:
    render_comparativo_tab(tab_comp)

with tab_fila:
    render_fila_tab(tab_fila)

with tab_dados:
    render_dados_tab(tab_dados, paths)
