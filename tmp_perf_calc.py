import time
from pathlib import Path

from src.logic.core.logic import (
    _load_with_version,
    merge_indicadores_from_faturamento,
    prepare_training_dataframe,
    clean_training_dataframe,
    _train_cached,
    gerar_resultados_modelos,
    calcular_intervalos_modelos,
)
from src.logic.data.buscaDeLojas import _ensure_loja_key, _get_loja_row


def load(name: str):
    base = Path("data")
    p = base / f"{name}.csv"
    st = p.stat()
    return _load_with_version(str(p), name, file_version=(st.st_mtime_ns, st.st_size))


def lap(msg: str, t0: float):
    dt = time.perf_counter() - t0
    print(f"{msg}: {dt:.3f}s")
    return time.perf_counter()


def main():
    t = time.perf_counter()
    dE = load("dEstrutura")
    dP = load("dPessoas")
    fF = load("fFaturamento2")
    fI = load("fIndicadores")
    fI = merge_indicadores_from_faturamento(fI, fF, dE)
    t = lap("load+merge_indicadores", t)

    train_df = prepare_training_dataframe(dE, dP, fI)
    train_df = clean_training_dataframe(train_df)
    t = lap("prepare+clean_train", t)

    train_n = _ensure_loja_key(train_df)
    row, _ = _get_loja_row(train_n, "NEVALDO ROCHA")
    features = row if row else train_df.iloc[0].to_dict()

    horas_disp = 44.0
    margem = 0.15
    anchor = 0.6

    cache_ver_hist = 9 + (6 * 1000)
    cache_ver_ideal = 9 + int(anchor * 100) + (6 * 1000) + 1234

    th = time.perf_counter()
    bundle_hist = _train_cached(train_df, "historico", horas_disp, margem, anchor_quantile=anchor, cache_version=cache_ver_hist)
    lap("train_cached_hist", th)

    ti = time.perf_counter()
    bundle_ideal = _train_cached(train_df, "ideal", horas_disp, margem, anchor_quantile=anchor, cache_version=cache_ver_ideal)
    lap("train_cached_ideal", ti)

    tg = time.perf_counter()
    gerar_resultados_modelos(
        bundle_hist,
        train_df,
        features,
        "historico",
        horas_disp,
        margem,
        algo_order=["catboost"],
        anchor_quantile=anchor,
        apply_cluster_blend=False,
        compute_metrics=False,
    )
    lap("resultados_hist_no_metrics", tg)

    tg2 = time.perf_counter()
    gerar_resultados_modelos(
        bundle_ideal,
        train_df,
        features,
        "ideal",
        horas_disp,
        margem,
        algo_order=["catboost"],
        anchor_quantile=anchor,
        apply_cluster_blend=False,
        compute_metrics=True,
        metrics_cache_bust=cache_ver_ideal,
    )
    lap("resultados_ideal_with_metrics", tg2)

    tc = time.perf_counter()
    calcular_intervalos_modelos(
        train_df,
        features,
        "historico",
        horas_disp,
        margem,
        ["catboost"],
        anchor_quantile=anchor,
        apply_cluster_blend=False,
    )
    lap("ci_hist", tc)

    tc2 = time.perf_counter()
    calcular_intervalos_modelos(
        train_df,
        features,
        "ideal",
        horas_disp,
        margem,
        ["catboost"],
        anchor_quantile=anchor,
        apply_cluster_blend=False,
    )
    lap("ci_ideal", tc2)


if __name__ == "__main__":
    main()
