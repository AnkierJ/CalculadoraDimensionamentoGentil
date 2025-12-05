#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilidades de clusterização (KMeans).

Se ainda não houver uso integrado, estas funções servem como ponto de extensão.
"""
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def treinar_kmeans(df_features: pd.DataFrame, n_clusters: Optional[int] = None) -> Optional[KMeans]:
    """Treina um KMeans simples nos dados fornecidos."""
    if df_features is None or df_features.empty:
        return None
    n_rows = len(df_features)
    if n_rows < 2:
        return None
    n_clusters = n_clusters or min(4, n_rows)
    if n_clusters < 2:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    model.scaler_ = scaler  # guardar scaler para previsão
    return model


def atribuir_cluster(df_features: pd.DataFrame, modelo_kmeans: Optional[KMeans]) -> pd.Series:
    """Atribui rótulos de cluster usando um modelo treinado (ou retorna zeros se não houver)."""
    if modelo_kmeans is None or df_features is None or df_features.empty:
        return pd.Series(0, index=df_features.index if df_features is not None else [])
    scaler = getattr(modelo_kmeans, "scaler_", None)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)
    else:
        X_scaled = scaler.transform(df_features)
    labels = modelo_kmeans.predict(X_scaled)
    return pd.Series(labels, index=df_features.index)
