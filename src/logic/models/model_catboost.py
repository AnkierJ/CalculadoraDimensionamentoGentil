from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    from catboost import CatBoostRegressor  # type: ignore
except ImportError:
    CatBoostRegressor = None


def _bool_to_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Converte campos booleanos em inteiros 0/1 para alimentar o modelo."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        if pd.api.types.is_bool_dtype(series):
            bool_ser = series.fillna(False)
        elif pd.api.types.is_numeric_dtype(series):
            bool_ser = series.fillna(0).astype(float).ne(0)
        else:
            s = series.astype(str).str.strip().str.upper()
            true_vals = {"VERDADEIRO", "SIM", "S", "TRUE", "T", "1", "YES", "Y"}
            false_vals = {"FALSO", "NAO", "N", "FALSE", "F", "0", "NO"}
            bool_ser = pd.Series(pd.NA, index=s.index, dtype="boolean")
            bool_ser = bool_ser.mask(s.isin(true_vals), True)
            bool_ser = bool_ser.mask(s.isin(false_vals), False)
            bool_ser = bool_ser.fillna(False)
        df[c] = bool_ser.astype(int)
    return df


class CatBoostQuantileModel:
    """Empacota o CatBoost central + quantis P5/P95, mantendo metadados das features."""

    def __init__(
        self,
        model_mid,
        model_low,
        model_high,
        feature_names: List[str],
        numeric_cols: List[str],
        categorical_cols: List[str],
        numeric_fill: Dict[str, float],
        categorical_fill: str,
    ) -> None:
        self.model_mid = model_mid
        self.model_low = model_low
        self.model_high = model_high
        self.model_feature_names_ = list(feature_names)
        self.numeric_features_ = list(numeric_cols)
        self.categorical_features_ = list(categorical_cols)
        self.numeric_fill_values_ = dict(numeric_fill)
        self.categorical_fill_value_ = categorical_fill
        self.is_catboost_quantile = True
        self.cluster_model_: Optional["KMeans"] = None

    def _prepare_features(self, feature_row: Dict[str, object]) -> pd.DataFrame:
        """Normaliza a linha de entrada para o CatBoost (unidades/valores iguais ao treino)."""
        row = {c: feature_row.get(c, None) for c in self.model_feature_names_}
        df = pd.DataFrame([row])
        df = df.reindex(columns=self.model_feature_names_)
        df = _bool_to_int(df, [c for c in ["Escritorio", "Copa", "Espaco Evento"] if c in df.columns])
        for col in self.numeric_features_:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                fill_val = self.numeric_fill_values_.get(col)
                df[col] = df[col].fillna(0.0 if fill_val is None else fill_val)
        for col in self.categorical_features_:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace({"nan": "", "None": ""})
                    .replace("", np.nan)
                    .fillna(self.categorical_fill_value_)
                )
        return df[self.model_feature_names_]

    def predict_from_row(self, feature_row: Dict[str, object]) -> float:
        df = self._prepare_features(feature_row)
        return float(self.model_mid.predict(df)[0])

    def predict_quantiles_from_row(self, feature_row: Dict[str, object]) -> tuple[float, float, float]:
        df = self._prepare_features(feature_row)
        low = float(self.model_low.predict(df)[0])
        mid = float(self.model_mid.predict(df)[0])
        high = float(self.model_high.predict(df)[0])
        return low, mid, high

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Compatibilidade basica com API sklearn (usa sempre o modelo central)."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.model_feature_names_)
        preds: List[float] = []
        for _, row in X.iterrows():
            preds.append(self.predict_from_row(row.to_dict()))
        return np.asarray(preds, dtype=float)


def _prepare_for_catboost(
    model,
    feature_row: Dict[str, object],
    feature_names: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    df = pd.DataFrame([{c: feature_row.get(c, None) for c in feature_names}])
    df = df.reindex(columns=feature_names)
    df = _bool_to_int(df, [c for c in ["Escritorio", "Copa", "Espaco Evento"] if c in df.columns])
    numeric_fill = getattr(model, "numeric_fill_values_", {}) or {}
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                fill_val = numeric_fill.get(col)
                df[col] = df[col].fillna(fill_val if fill_val is not None else 0.0)
    if categorical_cols:
        cat_fill = getattr(model, "categorical_fill_value_", "missing")
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "").replace("None", "")
                df[col] = df[col].replace("", np.nan).fillna(cat_fill)
    return df[feature_names]


def predict_catboost(model, feature_row: Dict[str, object]) -> Dict[str, float]:
    if model is None:
        raise ValueError("Modelo CatBoost nao treinado.")
    feature_names = (
        getattr(model, "model_feature_names_", None)
        or getattr(model, "feature_names_", None)
        or list(getattr(model, "feature_names_in_", []))
    )
    if not feature_names:
        raise ValueError("Modelo CatBoost sem feature_names_ configurado.")

    numeric_cols = getattr(model, "numeric_features_", []) or []
    categorical_cols = getattr(model, "categorical_features_", []) or []

    if isinstance(model, CatBoostQuantileModel):
        pred = model.predict_from_row(feature_row)
    elif CatBoostRegressor is not None and isinstance(model, CatBoostRegressor):
        df_model = _prepare_for_catboost(model, feature_row, feature_names, numeric_cols, categorical_cols)
        pred = float(model.predict(df_model)[0])
    else:
        # fallback para modelos legados/salvos com API compatvel
        df_model = _prepare_for_catboost(model, feature_row, feature_names, numeric_cols, categorical_cols)
        pred = float(model.predict(df_model)[0])

    return {"pred": float(pred)}


def train_catboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    used_features: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    sample_weights: pd.Series,
) -> Optional[CatBoostQuantileModel]:
    if CatBoostRegressor is None:
        raise ImportError("CatBoostRegressor indisponivel. Instale 'catboost'.")

    feature_order = [c for c in used_features if c in X.columns]
    if not feature_order:
        return None

    X_proc = X[feature_order].copy()
    numeric_in_model = [col for col in (numeric_cols or []) if col in X_proc.columns]
    categorical_in_model = [col for col in (categorical_cols or []) if col in X_proc.columns]
    numeric_fill_series = pd.Series(dtype="float64")
    if numeric_in_model:
        numeric_fill_series = X_proc[numeric_in_model].median(numeric_only=True).fillna(0.0)
        X_proc[numeric_in_model] = X_proc[numeric_in_model].fillna(numeric_fill_series)
    cat_fill_val = "missing"
    if categorical_in_model:
        for col in categorical_in_model:
            X_proc[col] = (
                X_proc[col]
                .astype(str)
                .replace({"nan": "", "None": ""})
                .replace("", np.nan)
                .fillna(cat_fill_val)
            )
    cat_feature_indices = [X_proc.columns.get_loc(col) for col in categorical_in_model]
    sample_weight_np = sample_weights.to_numpy(dtype=float)

    def _fit_cat_model(loss: str) -> CatBoostRegressor:
        model_cb = CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            n_estimators=800,
            subsample=0.9,
            l2_leaf_reg=5.0,
            loss_function=loss,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model_cb.fit(
            X_proc,
            y,
            cat_features=cat_feature_indices,
            sample_weight=sample_weight_np,
        )
        return model_cb

    model_mid = _fit_cat_model("RMSE")
    model_low = _fit_cat_model("Quantile:alpha=0.01")
    model_high = _fit_cat_model("Quantile:alpha=0.99")
    wrapper = CatBoostQuantileModel(
        model_mid=model_mid,
        model_low=model_low,
        model_high=model_high,
        feature_names=feature_order,
        numeric_cols=numeric_in_model,
        categorical_cols=categorical_in_model,
        numeric_fill=numeric_fill_series.to_dict(),
        categorical_fill=cat_fill_val,
    )
    return wrapper
