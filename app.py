import streamlit as st
from pathlib import Path

from src.logic.core.logic import _load_with_version, merge_indicadores_from_faturamento
from src.logic.utils.helpers import _standardize_cols
from src.views.calc_view import render_calc_tab
from src.views.comparativo_view import render_comparativo_tab
from src.views.dados_view import render_dados_tab
from src.views.diagnostics_view import render_diag_cache
from src.views.layout import inject_global_styles, render_header, render_tutorial, set_page_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Carregamento dos datasets em session_state (com recarga se arquivo mudar)
def _refresh_dataset(key: str, path: Path, schema: str) -> None:
    try:
        stat = path.stat()
        file_version = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        file_version = (0, 0)
    version_key = f"_version_{key}"
    if key not in st.session_state or st.session_state.get(version_key) != file_version:
        st.session_state[key] = _load_with_version(str(path), schema, file_version=file_version)
        st.session_state[version_key] = file_version


_refresh_dataset("dAmostras", DATA_DIR / "dAmostras.csv", "dAmostras")
_refresh_dataset("dEstrutura", DATA_DIR / "dEstrutura.csv", "dEstrutura")
_refresh_dataset("dPessoas", DATA_DIR / "dPessoas.csv", "dPessoas")
_refresh_dataset("fFaturamento2", DATA_DIR / "fFaturamento2.csv", "fFaturamento2")
_refresh_dataset("fIndicadores", DATA_DIR / "fIndicadores.csv", "fIndicadores")
st.session_state["_data_version"] = (
    st.session_state.get("_version_dAmostras"),
    st.session_state.get("_version_dEstrutura"),
    st.session_state.get("_version_dPessoas"),
    st.session_state.get("_version_fFaturamento2"),
    st.session_state.get("_version_fIndicadores"),
)

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

# Atualiza indicadores derivados a partir do fFaturamento2 e normaliza colunas
st.session_state["fIndicadores_raw"] = _standardize_cols(
    st.session_state.get("fIndicadores")
)
st.session_state["fIndicadores"] = merge_indicadores_from_faturamento(
    st.session_state.get("fIndicadores_raw"),
    st.session_state.get("fFaturamento2"),
    st.session_state.get("dEstrutura"),
)
st.session_state["fIndicadores"] = _standardize_cols(st.session_state["fIndicadores"])

# Tabs principais
tab_calc, tab_comp, tab_dados = st.tabs(["Cálculo", "Comparativo", "Dados de base"])

with tab_calc:
    render_calc_tab(tab_calc)

with tab_comp:
    render_comparativo_tab(tab_comp)

with tab_dados:
    render_dados_tab(tab_dados, paths)
