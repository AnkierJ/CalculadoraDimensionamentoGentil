import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.core.logic import (
    FEATURE_COLUMNS,
    _train_cached,
    collinearity_report,
    prepare_training_dataframe,
    clean_training_dataframe,
)


def render_diag_cache(tab_calc: DeltaGenerator) -> None:
    """Renderiza diagnóstico e limpeza de cache na aba de cálculo."""
    with tab_calc:
        st.subheader("Diagnóstico (opcional)")
        if st.button("Checar colinearidade das features"):
            train_df_diag = prepare_training_dataframe(
                st.session_state["dEstrutura"],
                st.session_state["dPessoas"],
                st.session_state["fIndicadores"],
            )
            train_df_diag = clean_training_dataframe(train_df_diag)
            if train_df_diag.empty:
                st.warning("Sem dados suficientes para avaliar colinearidade.")
            else:
                feat_disp = [c for c in FEATURE_COLUMNS if c in train_df_diag.columns]
                st.caption(f"{len(train_df_diag)} lojas válidas | {len(feat_disp)} variáveis analisadas.")
                rep = collinearity_report(train_df_diag, FEATURE_COLUMNS, corr_thr=0.85)
                st.write("🔁 Pares altamente correlacionados (|r|≥0.85):")
                if rep["high_corr_pairs"]:
                    st.dataframe(pd.DataFrame(rep["high_corr_pairs"], columns=["col1", "col2", "r"]), use_container_width=True)
                else:
                    st.success("Nenhum par acima do limiar.")
                st.write("📐 VIF por coluna (ideal < 10):")
                st.dataframe(pd.DataFrame(rep["vif"], columns=["col", "VIF"]), use_container_width=True)

        st.subheader("Re-treinar modelo (opcional)")
        if st.button("🔄 Re-treinar (limpar cache)"):
            _train_cached.clear()
            st.success("Cache limpo. O próximo cálculo irá re-treinar o modelo.")
