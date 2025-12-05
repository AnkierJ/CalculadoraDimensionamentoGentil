from pathlib import Path

from src.logic.core.logic import clean_training_dataframe, load_csv_path, prepare_training_dataframe
from src.logic.utils.helpers import (
    get_schema_dEstrutura,
    get_schema_dPessoas,
    get_schema_fIndicadores,
)

base = Path(r"C:/Users/ankier.lima/Gentil Negócios/File server-GN - Comercial/Comercial 360/01. Dimensionamento do Time de Venda Direta/03. Calculadora")
dEstrutura = load_csv_path(str(base / "dEstrutura.csv"), get_schema_dEstrutura())
dPessoas = load_csv_path(str(base / "dPessoas.csv"), get_schema_dPessoas())
fIndicadores = load_csv_path(str(base / "fIndicadores.csv"), get_schema_fIndicadores())
train_df = prepare_training_dataframe(dEstrutura, dPessoas, fIndicadores)
train_df = clean_training_dataframe(train_df)
print("Columns:", train_df.columns.tolist())
if "Faturamento/Hora" in train_df.columns:
    print(train_df[["Loja", "Faturamento/Hora", "QtdAux"]].head())
    print("Faturamento/Hora missing", train_df["Faturamento/Hora"].isna().sum(), "rows", len(train_df))
else:
    print("Faturamento/Hora not found")
