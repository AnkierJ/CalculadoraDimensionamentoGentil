# Aba "Calculadora" – Fluxo e pontos de integração

## 1. Resumo da experiência do usuário
- A aba recebe os uploads já carregados (`dEstrutura`, `dPessoas`, `fIndicadores`, etc.) e mostra os cards principais (títulos, lookup da loja, configuração de parâmetros).
- Possui um formulário para pesquisa de loja (por código BCPS/SAP ou por nome). O resultado alimenta `st.session_state["lookup_row"]`.
- Há uma seção de configuração operacional (dias úteis, horas operacionais, buffers). Esses valores são lidos diretamente por `app.py` e repassados ao pipeline em `src/logic.py`.
- Após a submissão (“Calcular Qtd Auxiliares”), o app chama:
  - `prepare_training_dataframe` → consolida dados históricos.
  - `train_all_auxiliares_models` → prepara/treina CatBoost e XGBoost.
  - `predict_qtd_auxiliares`/`predict_with_uncertainty` → gera previsões e ICs.
- A saída compara os três blocos: CatBoost, XGBoost e Teoria das Filas (headcount contínuo, rhos e deltas).

## 2. Dependências principais
- Para renderizar métricas o app usa funções de `logic.py`: `evaluate_model_cv`, `predict_qtd_auxiliares`, `predict_with_uncertainty`, `_format_queue_diag`, etc.
- O bloco de fila depende dos helpers de fila (`estimate_queue_inputs`, `modelo_fila`) e das features operacionais geradas no `logic`.
- O formulário de upload reutiliza `_load_with_version`, `prepare_training_dataframe`, `clean_training_dataframe`.

## 3. Possível reestruturação (apenas documental)
- Extrair toda a lógica de renderização em classes/helpers separados, por exemplo:
  - `ui.calculadora.lookup` → componentes da pesquisa.
  - `ui.calculadora.metrics` → renderização dos cards de métricas, comparativos e CI.
- Consolidar os textos fixos (legendas, mensagens de erro) em um módulo único (`ui/copy.py`), reduzindo duplicação.
- Documentar as interações com `st.session_state` (nome das chaves e propósito) em uma tabela anexa ao documento.

