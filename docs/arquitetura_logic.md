# Arquitetura proposta para `src/logic`

> Documento apenas documental. Nenhuma função foi movida até o momento; este guia descreve como reorganizá-las.

## 1. Visão atual (resumo)
- `logic.py` concentra:
  - Preparação de dados (`prepare_training_dataframe`, `_apply_operational_derived_features`, `clean_training_dataframe`).
  - Treino/predict de ML (`train_auxiliares_model`, `train_all_auxiliares_models`, `predict_qtd_auxiliares`, `predict_with_uncertainty`).
  - Funções do modo ideal/simplificado (`calcular_qtd_aux_ideal`, `ideal_simplificado_por_fluxo`, etc.).
  - Utilidades de fila, lookup de lojas, diagnósticos e validações.
- Essa centralização torna o arquivo difícil de manter (~2.000+ linhas).

## 2. Organização sugerida
```
src/
└── logic/
    ├── __init__.py                # exports públicos
    ├── machine_learning/
    │   ├── __init__.py
    │   ├── historico.py           # treino/predict specifico do modo histórico
    │   ├── ideal.py               # variações do modo ideal (pesos, target)
    │   └── features.py            # engenharia de features operacionais e fila
    ├── simplificado/
    │   ├── __init__.py
    │   └── calculos.py            # funções do modo “ideal simplificado/manual”
    ├── queues.py                  # teoria das filas (modelo, diagnósticos)
    ├── lookup.py                  # pesquisa de loja, normalização de códigos
    ├── diagnostics.py             # métricas cruzadas, importâncias, logs
    └── data_ingest.py             # leitura/validação dos datasets
```

### Exemplos de alocação
- `machine_learning/historico.py`
  - `prepare_training_dataframe_historico`
  - `train_all_auxiliares_models`, `predict_qtd_auxiliares`
  - helpers exclusivos (ex.: `evaluate_model_cv`)
- `machine_learning/ideal.py`
  - funções que usam `mode == "ideal"` (sample weights, targets).
- `machine_learning/features.py`
  - `_apply_operational_derived_features`, `_compute_queue_features`, `_assign_clusters`.
  - Factory/registries para `monotone_constraints`.
- `simplificado/calculos.py`
  - `ideal_simplificado_por_fluxo`, `calcular_qtd_aux_ideal`, `carga_total_horas_loja`.
- `queues.py`
  - `estimate_queue_inputs`, `modelo_fila`, helpers de debug.
- `lookup.py`
  - `_filter_df_by_loja`, `_get_loja_row`, `_norm_code`.
- `diagnostics.py`
  - `_format_queue_diag`, `render_diagnostics`, importâncias de features.
- `data_ingest.py`
  - `_standardize_cols`, `_load_with_version`, `read_csv_with_schema`.

## 3. Benefícios esperados
- Arquivos menores e com responsabilidade única.
- Facilita testes unitários (cada módulo pode receber testes específicos).
- Reuso explícito das features derivadas tanto no treino quanto no predict.
- Clareza na camada de UI (o `app.py` passa a importar funções mais coesas).

## 4. Próximos passos recomendados
1. Criar os diretórios e mover as funções listadas (sem alterar assinatura).
2. Atualizar `app.py` e demais módulos para usar os novos caminhos.
3. Adicionar testes básicos por módulo (e.g., `tests/test_lookup.py`, `tests/test_queues.py`).
4. Atualizar `README.md` apontando para este documento e descrevendo o novo mapa.

