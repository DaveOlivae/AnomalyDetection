"""
tune_random_forest.py
═════════════════════
Busca de hiperparâmetros para o Random Forest no dataset TEP.

Fluxo:
  1. Carrega e prepara os dados com TEPLoader  (data_loader.py)
  2. Normaliza com StandardScaler
  3. Executa RandomizedSearchCV no conjunto train (usando val como fold fixo)
  4. Re-treina o melhor modelo em train + val
  5. Avalia no test  com evaluation.py  (métricas + gráficos completos)
  6. Salva o modelo final e os resultados

Uso:
    python tune_random_forest.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

from src.binary_classification.time_series.time_series_rft_finetune.data_loader import TEPLoader, LoaderConfig
from src.binary_classification.time_series.time_series_rft_finetune.evaluation import (
    full_evaluation,
    plot_metrics_comparison,
    plot_search_results,
    save_metrics_csv,
    save_model,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← ajuste aqui
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path("tep_results_rf")
RANDOM_STATE = 42
N_ITER       = 30       # número de combinações testadas no RandomizedSearch
CV_JOBS      = -1       # paralelismo (-1 = todos os núcleos)
SCORER       = "f1"     # métrica guia da busca

# Espaço de busca de hiperparâmetros
PARAM_DIST = {
    "n_estimators":      randint(100, 600),        # número de árvores
    "max_depth":         [None, 5, 10, 15, 20, 30],# profundidade máxima
    "min_samples_split": randint(2, 30),            # mínimo de amostras para dividir
    "min_samples_leaf":  randint(1, 20),            # mínimo de amostras na folha
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight":      [None, "balanced"],        # peso das classes
    "bootstrap":         [True, False],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_sep(title: str = "") -> None:
    line = "═" * 65
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def _print_params(params: dict, indent: int = 4) -> None:
    pad = " " * indent
    for k, v in sorted(params.items()):
        print(f"{pad}{k:<25}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    _print_sep("1/5  CARREGANDO DADOS")

    cfg    = LoaderConfig()                 # ← edite aqui se precisar mudar paths
    loader = TEPLoader(cfg)
    data   = loader.load_all()

    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]
    feat_names       = data["feat_names"]

    # ── 2. Normalização ───────────────────────────────────────────────────────
    _print_sep("2/5  NORMALIZANDO")

    #  Nota: Random Forest não precisa de normalização para o treino em si,
    #  mas normalizamos para que os dados sejam comparáveis a outros modelos
    #  e para o caso de querermos empilhar um scaler no pipeline futuro.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    save_model(scaler, OUTPUT_DIR / "scaler.joblib")

    # ── 3. Baseline (parâmetros padrão) ───────────────────────────────────────
    _print_sep("3/5  BASELINE RANDOM FOREST")

    baseline_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("  Treinando baseline ...")
    t_bl = time.time()
    baseline_rf.fit(X_train, y_train)
    print(f"  Baseline treinado em {time.time() - t_bl:.1f}s")

    baseline_metrics = full_evaluation(
        baseline_rf, "RF Baseline",
        X_val, y_val,
        feat_names=feat_names,
        out_dir=OUTPUT_DIR / "baseline",
    )

    # ── 4. RandomizedSearchCV ─────────────────────────────────────────────────
    _print_sep("4/5  RANDOMIZED SEARCH CV")

    #  Usamos PredefinedSplit para tratar val como o único fold de CV.
    #  Isso garante que a busca veja val mas nunca veja test.
    #
    #  PredefinedSplit convention:
    #    -1 → amostra usada para treino no fold
    #     0 → amostra usada para validação no fold
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    split_index = np.concatenate([
        np.full(len(X_train), -1),   # treino
        np.full(len(X_val),    0),   # validação
    ])
    pds = PredefinedSplit(split_index)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring=SCORER,
        cv=pds,
        n_jobs=CV_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True,
        refit=False,    # re-treinaremos manualmente em train+val
    )

    print(f"\n  Iniciando busca: {N_ITER} iterações | scorer={SCORER}")
    t_search = time.time()
    search.fit(X_trainval, y_trainval)
    print(f"\n  Busca concluída em {time.time() - t_search:.1f}s")

    # Salva resultados da busca
    cv_df = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    cv_df.to_csv(OUTPUT_DIR / "search_results.csv", index=False)
    print(f"\n  Top 5 combinações encontradas:")
    show_cols = (
        ["mean_test_score", "std_test_score"]
        + [c for c in cv_df.columns if c.startswith("param_")]
    )
    print(cv_df[show_cols].head(5).to_string(index=False))

    best_params = search.best_params_
    print(f"\n  ★ Melhores hiperparâmetros (val F1={search.best_score_:.4f}):")
    _print_params(best_params)

    # Plot do espaço de busca (n_estimators como exemplo)
    plot_search_results(
        search.cv_results_,
        param_name="param_n_estimators",
        score_name="mean_test_score",
        out_path=OUTPUT_DIR / "search_n_estimators.png",
    )

    # ── 5. Re-treino em train+val com os melhores params ─────────────────────
    _print_sep("5/5  MODELO FINAL (TRAIN + VAL) → AVALIAÇÃO NO TEST")

    tuned_rf = RandomForestClassifier(
        **best_params,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("  Re-treinando modelo tunado em train + val ...")
    t_final = time.time()
    tuned_rf.fit(X_trainval, y_trainval)
    print(f"  Treinado em {time.time() - t_final:.1f}s")

    tuned_metrics = full_evaluation(
        tuned_rf, "RF Tunado",
        X_test, y_test,
        feat_names=feat_names,
        out_dir=OUTPUT_DIR / "tuned",
    )

    # ── Relatório comparativo baseline vs tunado ──────────────────────────────
    _print_sep("RELATÓRIO FINAL — Baseline vs Tunado")

    results = {
        "RF Baseline": baseline_metrics,
        "RF Tunado":   tuned_metrics,
    }
    results_df = save_metrics_csv(results, OUTPUT_DIR / "metrics_comparison.csv")

    print("\n" + results_df.to_string(col_space=18))

    plot_metrics_comparison(
        results_df,
        out_path=OUTPUT_DIR / "metrics_comparison.png",
    )

    # Salva modelo final
    save_model(tuned_rf, OUTPUT_DIR / "rf_tuned.joblib")

    # ── Sumário ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_sep(f"✓  Pipeline concluído em {elapsed:.1f}s")
    print(f"  Resultados em: {OUTPUT_DIR.resolve()}/")
    print(f"\n  Arquivos gerados:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(OUTPUT_DIR)}")
    print()


if __name__ == "__main__":
    main()
