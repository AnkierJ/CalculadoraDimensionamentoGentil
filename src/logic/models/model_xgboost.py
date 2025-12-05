from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor  # type: ignore
except ImportError:
    XGBRegressor = None


def _build_xgb_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """Pre-processador especifico para o XGBoost (numerico + OneHot para categoricas)."""
    transformers = []
    if numeric_cols:
        transformers.append(
            ("numericas", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_cols)
        )
    if categorical_cols:
        encoder_args: Dict[str, object] = {"handle_unknown": "ignore"}
        if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
            encoder_args["sparse_output"] = False
        else:
            encoder_args["sparse"] = False
        transformers.append(
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(**encoder_args)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _create_xgb_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    categorical_cardinality: Optional[Dict[str, int]] = None,
) -> Pipeline:
    """Cria o pipeline completo (pre-processamento + estimador XGBoost)."""
    if XGBRegressor is None:
        raise ImportError("XGBoostRegressor indisponivel. Instale 'xgboost'.")
    preprocessor = _build_xgb_preprocessor(numeric_cols, categorical_cols)
    monotone_str = None
    estimator = XGBRegressor(
        n_estimators=320,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        gamma=0.15,
        objective="reg:squarederror",
        reg_alpha=0.4,
        reg_lambda=1.2,
        random_state=42,
        tree_method="hist",
        monotone_constraints=monotone_str,
    )
    return Pipeline(
        steps=[
            ("preprocessamento", preprocessor),
            ("modelo", estimator),
        ]
    )


def predict_xgboost(model, feature_row: Dict[str, object]) -> Dict[str, float]:
    if model is None:
        raise ValueError("Modelo XGBoost nao treinado.")
    feature_names = (
        getattr(model, "model_feature_names_", None)
        or getattr(model, "feature_names_", None)
        or list(getattr(model, "feature_names_in_", []))
    )
    if not feature_names:
        raise ValueError("Modelo XGBoost sem feature_names_ configurado.")
    df = pd.DataFrame([{c: feature_row.get(c, None) for c in feature_names}])
    df = df.reindex(columns=feature_names)
    numeric_cols = getattr(model, "numeric_features_", []) or []
    categorical_cols = getattr(model, "categorical_features_", []) or []
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    fill_vals = getattr(model, "fill_values_", None)
    if fill_vals is not None:
        df = df.fillna(fill_vals)
    pred = float(model.predict(df)[0])
    return {"pred": pred}


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    used_features: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    sample_weights: pd.Series,
    categorical_cardinality: Optional[Dict[str, int]] = None,
) -> Optional[Pipeline]:
    if XGBRegressor is None:
        raise ImportError("XGBoostRegressor indisponivel. Instale 'xgboost'.")

    model = _create_xgb_pipeline(numeric_cols, categorical_cols, categorical_cardinality)
    fit_params = {"modelo__sample_weight": sample_weights.to_numpy(dtype=float)}
    model.fit(X, y, **fit_params)
    model.model_feature_names_ = list(used_features)
    model.numeric_features_ = list(numeric_cols)
    model.categorical_features_ = list(categorical_cols)
    return model
