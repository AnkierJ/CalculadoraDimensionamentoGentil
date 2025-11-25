# 🧮 Calculadora de Dimensionamento do Comercial — MVP

Uma aplicação Streamlit simples para estimar o tamanho ideal de equipe comercial baseado em frequências e tempos médios de atendimento.

## 🚀 Como Executar (Resolvendo Problemas de Firewall)

### Opção 1: Execução Automática (Recomendada)
1. **Duplo clique** no arquivo `executar_calculadora.bat`
2. Aguarde a instalação das dependências (se necessário)
3. A aplicação abrirá automaticamente em `http://localhost:8501`

### Opção 2: Execução Manual
1. Abra o terminal/prompt de comando na pasta do projeto
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script de configuração:
   ```bash
   python run_app.py
   ```

### Opção 3: Execução Direta (se as opções acima não funcionarem)
```bash
streamlit run app.py --server.address localhost --server.port 8501 --server.headless true
```

## 🔧 Resolução de Problemas de Firewall

Se você encontrar problemas de firewall:

1. **Windows Defender**: Adicione exceção para Python e Streamlit
2. **Antivírus**: Configure exceções para a pasta do projeto
3. **Firewall Corporativo**: Use as configurações de localhost (127.0.0.1:8501)

## 📋 Funcionalidades

- **Cálculo Manual**: Insira quantidade de atendimentos e tempo médio
- **Upload CSV**: Carregue dados de atividades em lote
- **Configurações Flexíveis**: Ajuste margem de folga e fator de monotonia
- **Resultados Detalhados**: Visualize carga total e equipe necessária

## 📁 Estrutura do Projeto

```
calculadora_comercial/
├── app.py                    # Aplicação principal
├── src/logic.py             # Lógica de cálculo
├── data/exemplo.csv         # Exemplo de dados
├── run_app.py              # Script de execução segura
├── executar_calculadora.bat # Execução automática (Windows)
├── .streamlit/config.toml   # Configurações do Streamlit
└── requirements.txt         # Dependências
```

## 🛠️ Dependências

- streamlit==1.39.0
- pandas>=2.0.0
- numpy>=1.25.0

## 📊 Como Usar

1. **Entrada Manual**: Preencha quantidade de atendimentos e tempo médio
2. **Upload CSV** (opcional): Envie arquivo com colunas: `atividade`, `frequencia`, `tempo_min`
3. **Configure**: Ajuste margem de folga e fator de monotonia
4. **Calcule**: Clique em "Calcular dimensionamento"

## 🔒 Segurança

A aplicação está configurada para rodar apenas em localhost (127.0.0.1) para máxima segurança e compatibilidade com firewalls corporativos.
