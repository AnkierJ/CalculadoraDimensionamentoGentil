# Aba "Dados de Base" – Upload, validação e preparo

## 1. Objetivo
Centralizar a documentação do fluxo de ingestão dos arquivos `dAmostras`, `dEstrutura`, `dPessoas`, `fIndicadores`, `fFaturamento2`. Essa aba garante que o usuário entenda quais colunas são obrigatórias e como os dados são usados no restante da calculadora.

## 2. Fluxo atual
1. **Upload / arraste-e-solte**: cada dataset tem um componente `st.file_uploader`.
2. **Validação estrutural**: `read_csv_with_schema`, `validate_df` conferem se as colunas obrigatórias existem e fazem conversão de tipos básicos.
3. **Normalização**: `_standardize_cols`, `_standardize_row` tratam nomes, acentos, strings vazias.
4. **Persistência em sessão**: cada dataframe validado vai para `st.session_state["<nome>"]`.
5. **Uso posterior**:
   - `prepare_training_dataframe` consome `dEstrutura`, `dPessoas`, `fIndicadores`.
   - `clean_training_dataframe` realiza as últimas checagens de outliers/nulos.
   - `estimate_cluster_indicators`, `estimate_process_frequencies_from_indicadores` complementam informações quando campos estão ausentes.

## 3. Conteúdo mínimo por arquivo
- **dEstrutura**: identificação da loja (`Loja`, `BCPS`, `SAP`), atributos físicos (área, caixas, horários), alocação atual de auxiliares.
- **dPessoas**: `QtdAux`, disponibilidade, eventuais campos de absenteísmo.
- **fIndicadores**: métricas comerciais (Base Ativa, Receita, Pedidos, I4–I6, etc.).
- **dAmostras** / **fFaturamento2**: usados para diagnósticos e inferência de tempos médios quando ausentes.

## 4. Sugestões de organização adicional
- Criar um `docs/esquema_dados.md` que liste coluna por coluna, indicando:
  - Tipo (`numérico`, `categorias`, `bool`).
  - Obrigatoriedade.
  - Funções que a utilizam (por ex.: `DiasOperacionais` → `prepare_training_dataframe`, `modelo_fila`).
- Extrair para `src/data/ingest.py` os utilitários de leitura/validação (hoje espalhados em `logic.py`), deixando `app.py` responsável apenas pela camada de UI.
