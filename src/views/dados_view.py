from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.logic.core.logic import (
    get_schema_dAmostras,
    get_schema_dEstrutura,
    get_schema_dPessoas,
    get_schema_fFaturamento,
    get_schema_fIndicadores,
    render_append,
)


def render_dados_tab(tab_dados: DeltaGenerator, paths: Dict[str, Path]) -> None:
    with tab_dados:
        st.subheader("Dados de base")
        aba_preview, aba_upload = st.tabs(["Base atual (somente leitura)", "Adicionar dados (upload)"])
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
            with st.expander("fFaturamento"):
                st.dataframe(st.session_state["fFaturamento"].head(10), use_container_width=True)
                path = paths.get("fFaturamento")
                if path and path.exists():
                    st.write(f"📦 Tamanho de fFaturamento.csv: {path.stat().st_size:,} bytes")
                else:
                    st.warning("Arquivo fFaturamento.csv não encontrado.")
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
                for nome in ["dAmostras", "dEstrutura", "dPessoas", "fFaturamento", "fIndicadores"]:
                    df = st.session_state[nome]
                    csv_bytes = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
                    st.download_button(
                        label=f"⬇️ Baixar {nome}.csv",
                        data=csv_bytes,
                        file_name=f"{nome}.csv",
                        mime="text/csv",
                        use_container_width=True,
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
