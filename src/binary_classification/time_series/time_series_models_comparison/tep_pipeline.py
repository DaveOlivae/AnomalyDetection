#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
  Tennessee Eastman Process — Binary Fault Detection Pipeline
════════════════════════════════════════════════════════════════════════════════

  Estrutura do dataset esperada (4 CSVs):
    TEP_FaultFree_Training.csv  │  TEP_Faulty_Training.csv
    TEP_FaultFree_Testing.csv   │  TEP_Faulty_Testing.csv

  Colunas esperadas: faultNumber, simulationRun, sample + 52 variáveis de processo

  Instalação das dependências:
    pip install pandas numpy scikit-learn lightgbm matplotlib seaborn

  Uso:
    python tep_pipeline.py
    (ajuste DATA_DIR e os paths dos arquivos na seção CONFIG abaixo)
════════════════════════════════════════════════════════════════════════════════
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[AVISO] LightGBM não encontrado — modelo LGBM será pulado.")
    print("        Instale com: pip install lightgbm\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← ajuste aqui
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("../data/raw/")                              # pasta com os 4 CSVs

FF_TRAIN = DATA_DIR / "TEP_FaultFree_Training.csv"
FA_TRAIN = DATA_DIR / "TEP_Faulty_Training.csv"
FF_TEST  = DATA_DIR / "TEP_FaultFree_Testing.csv"
FA_TEST  = DATA_DIR / "TEP_Faulty_Testing.csv"

WINDOW_SIZE  = 20    # tamanho da janela deslizante (amostras)
STRIDE       = 5     # passo entre janelas (overlap de 75%)
RANDOM_STATE = 42

# ── Orçamento de memória ───────────────────────────────────────────────────────
#  Treino  → fault-free: 500 rows/sim  |  faulty: 20×500 = 10 000 rows/sim
#  Teste   → fault-free: 960 rows/sim  |  faulty: 20×960 = 19 200 rows/sim
#
#  Para 50/50, precisamos de ~20x mais sims fault-free do que faulty.
#  Valores abaixo geram ~120k janelas de treino e ~56k de teste (bem leve).
#
#    Treino: 120 ff sims → 60 000 rows  |  6 fa sims  → 60 000 rows
#    Teste:   60 ff sims → 57 600 rows  |  3 fa sims  → 57 600 rows
#
#  Diminua esses valores se ainda estiver sem memória.
N_FF_TRAIN = 120
N_FA_TRAIN = 6
N_FF_TEST  = 60
N_FA_TEST  = 3

OUTPUT_DIR = Path("tep_results")
OUTPUT_DIR.mkdir(exist_ok=True)

META_COLS = ["faultNumber", "simulationRun", "sample"]


# ══════════════════════════════════════════════════════════════════════════════
#  1.  CARREGAMENTO & AMOSTRAGEM
# ══════════════════════════════════════════════════════════════════════════════
def _sample_sims(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Amostra n simulações completas (por simulationRun), sem quebrar séries."""
    all_sims = df["simulationRun"].unique()
    chosen   = rng.choice(all_sims, size=min(n, len(all_sims)), replace=False)
    return df[df["simulationRun"].isin(chosen)].reset_index(drop=True)


def load_split(
    ff_path: Path, fa_path: Path,
    n_ff: int, n_fa: int,
    rng: np.random.RandomState,
    split_name: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Carrega e amostra um split (treino ou teste).
    Retorna DataFrame balanceado (50/50) com coluna 'label'.
    """
    print(f"\n  [{split_name}] Carregando fault-free: {ff_path.name}")
    ff = pd.read_csv(ff_path)
    print(f"    → {len(ff):>10,} linhas | {ff['simulationRun'].nunique()} sims")

    print(f"  [{split_name}] Carregando faulty:     {fa_path.name}")
    fa = pd.read_csv(fa_path)
    print(f"    → {len(fa):>10,} linhas | {fa['simulationRun'].nunique()} sims")

    ff_sub = _sample_sims(ff, n_ff, rng)
    fa_sub = _sample_sims(fa, n_fa, rng)
    del ff, fa  # libera memória imediatamente

    ff_sub["label"] = 0
    fa_sub["label"] = 1  # todas as falhas viram label=1 (binário)

    df = pd.concat([ff_sub, fa_sub], ignore_index=True)
    del ff_sub, fa_sub

    feat_cols = [c for c in df.columns if c not in META_COLS + ["label"]]
    dist = df["label"].value_counts().sort_index().to_dict()
    print(f"  [{split_name}] Amostrado → {dist} | {len(feat_cols)} features")
    return df, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
#  2.  SLIDING WINDOW → FEATURES ESTATÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════
def _window_features(window: np.ndarray) -> np.ndarray:
    """
    Extrai features de uma janela (window_size × n_vars):
      média, desvio padrão, mínimo, máximo por variável
      → 4 × n_vars features
    """
    return np.concatenate([
        window.mean(axis=0),
        window.std(axis=0),
        window.min(axis=0),
        window.max(axis=0),
    ])


def _windows_from_group(
    group: pd.DataFrame,
    feat_cols: List[str],
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica janela deslizante numa série temporal de uma única simulação/falha.
    Label da janela = maioria dos labels no intervalo (trivial: todos iguais).
    """
    grp = group.sort_values("sample")
    vals   = grp[feat_cols].values.astype(np.float32)  # (T, n_vars)
    labels = grp["label"].values                        # (T,)
    T      = len(vals)

    X_list, y_list = [], []
    for start in range(0, T - window + 1, stride):
        w   = vals[start : start + window]
        lbl = labels[start : start + window]
        X_list.append(_window_features(w))
        y_list.append(int(lbl.mean() >= 0.5))

    if not X_list:
        return np.empty((0, 4 * len(feat_cols))), np.empty(0, dtype=int)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)


def build_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    window: int,
    stride: int,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Itera sobre cada série temporal (simulationRun × faultNumber) e gera janelas.
    Simulações nunca são quebradas entre treino e teste.
    """

    print(f"\n  Gerando janelas [{desc}]  (window={window}, stride={stride}) ...")

    Xs, ys = [], []

    # Agrupa por (simulationRun, faultNumber) — cada grupo é uma série temporal
    for _, grp in df.groupby(["simulationRun", "faultNumber"], sort=False):
        x, y = _windows_from_group(grp, feat_cols, window, stride)
        if len(x) > 0:
            Xs.append(x)
            ys.append(y)

    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(np.int8)
    counts = np.bincount(y.astype(int))
    print(f"    → {X.shape[0]:,} janelas | Normal={counts[0]:,} | Falha={counts[1]:,} "
          f"| shape={X.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  3.  DEFINIÇÃO DOS MODELOS
# ══════════════════════════════════════════════════════════════════════════════
def build_models() -> Dict:
    models = {
        # ── Baseline linear ──────────────────────────────────────────────────
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="saga",
            max_iter=500,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        # ── Ensemble de árvores ──────────────────────────────────────────────
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        # ── Rede neural (MLP) ────────────────────────────────────────────────
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=100,
            batch_size=512,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ),

        # ── SVM linear calibrado ─────────────────────────────────────────────
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(
                C=0.5,
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            cv=3,
        ),
    }

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    return models


# ══════════════════════════════════════════════════════════════════════════════
#  4.  TREINO & AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    name: str,
    model,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> Tuple[Dict, np.ndarray]:

    print(f"\n  ▶ {name}")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"    Treino concluído em {train_time:.1f}s")

    y_pred = model.predict(X_te)

    # AUC-ROC (precisa de probabilidades)
    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)
    except AttributeError:
        auc = float("nan")

    metrics = {
        "Acurácia":          accuracy_score(y_te, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_te, y_pred),
        "Precisão":          precision_score(y_te, y_pred, zero_division=0),
        "Recall":            recall_score(y_te, y_pred, zero_division=0),
        "F1":                f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC":           auc,
        "Tempo treino (s)":  round(train_time, 2),
    }
    cm = confusion_matrix(y_te, y_pred)

    print(f"    F1={metrics['F1']:.4f}  |  AUC={metrics['ROC-AUC']:.4f}  "
          f"|  Recall={metrics['Recall']:.4f}")
    print(classification_report(
        y_te, y_pred,
        target_names=["Normal (0)", "Falha  (1)"],
        digits=4,
    ))
    return metrics, cm


# ══════════════════════════════════════════════════════════════════════════════
#  5.  RELATÓRIOS & VISUALIZAÇÕES
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(cms: Dict, out_dir: Path) -> None:
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Falha"],
            yticklabels=["Normal", "Falha"],
            ax=ax, linewidths=0.5,
        )
        # Anotações percentuais
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.72,
                        f"({cm[i,j]/total*100:.1f}%)",
                        ha="center", va="center",
                        fontsize=8, color="grey")
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

    fig.suptitle(
        "Matrizes de Confusão — TEP Binary Classification",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_metrics_comparison(results_df: pd.DataFrame, out_dir: Path) -> None:
    cols  = ["Acurácia", "Precisão", "Recall", "F1", "ROC-AUC"]
    avail = [c for c in cols if c in results_df.columns]
    data  = results_df[avail].astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 2), 5))
    x    = np.arange(len(data))
    w    = 0.15
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(avail)))

    for i, (col, color) in enumerate(zip(avail, colors)):
        bars = ax.bar(x + i * w, data[col], w, label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w * (len(avail) - 1) / 2)
    ax.set_xticklabels(data.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Comparação de Modelos — TEP Binary Classification",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_feature_importance(model, feat_names: List[str],
                            model_name: str, out_dir: Path,
                            top_n: int = 20) -> None:
    """Plot feature importance para modelos que suportam (RF, LGBM)."""
    try:
        imp = model.feature_importances_
    except AttributeError:
        return

    idx     = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    ax.barh(np.array(feat_names)[idx], imp[idx], color="steelblue")
    ax.set_xlabel("Importância")
    ax.set_title(f"Top {top_n} Features — {model_name}", fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    path = out_dir / f"feature_importance_{safe_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    t_start = time.time()
    rng     = np.random.RandomState(RANDOM_STATE)

    # ── 1. Dados ─────────────────────────────────────────────────────────────
    sep = "═" * 65
    print(f"\n{sep}")
    print("  1/5  CARREGANDO DADOS")
    print(sep)

    train_df, feat_cols = load_split(
        FF_TRAIN, FA_TRAIN, N_FF_TRAIN, N_FA_TRAIN, rng, "TREINO"
    )
    test_df, _ = load_split(
        FF_TEST, FA_TEST, N_FF_TEST, N_FA_TEST, rng, "TESTE"
    )

    # ── 2. Sliding windows ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  2/5  GERANDO JANELAS DESLIZANTES")
    print(sep)

    X_train, y_train = build_windows(train_df, feat_cols, WINDOW_SIZE, STRIDE, "treino")
    X_test,  y_test  = build_windows(test_df,  feat_cols, WINDOW_SIZE, STRIDE, "teste")
    del train_df, test_df  # libera RAM

    # Nomes das features para importância
    stat_names = ["mean", "std", "min", "max"]
    feat_names = [f"{s}_{c}" for s in stat_names for c in feat_cols]

    # ── 3. Normalização ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  3/5  NORMALIZANDO (StandardScaler no treino)")
    print(sep)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"  Features: {X_train.shape[1]}  |  "
          f"Treino: {X_train.shape[0]:,} janelas  |  "
          f"Teste: {X_test.shape[0]:,} janelas")

    # ── 4. Treino & Avaliação ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  4/5  TREINANDO MODELOS")
    print(sep)

    models  = build_models()
    results = {}
    cms     = {}

    for name, model in models.items():
        m, cm        = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results[name] = m
        cms[name]     = cm

    # ── 5. Relatório ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  5/5  RELATÓRIO FINAL")
    print(sep)

    results_df = (
        pd.DataFrame(results)
        .T
        .sort_values("F1", ascending=False)
        .round(4)
    )

    print("\n  ┌─ RESUMO (ordenado por F1) " + "─" * 37)
    print(results_df.to_string(col_space=18))
    print()

    results_df.to_csv(OUTPUT_DIR / "metrics.csv")
    print(f"\n  Tabela salva: {OUTPUT_DIR / 'metrics.csv'}")

    print("\n  Gerando visualizações ...")
    plot_confusion_matrices(cms, OUTPUT_DIR)
    plot_metrics_comparison(results_df, OUTPUT_DIR)

    for name, model in models.items():
        plot_feature_importance(model, feat_names, name, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\n{sep}")
    print(f"  ✓  Pipeline concluído em {elapsed:.1f}s")
    print(f"  ✓  Resultados salvos em: {OUTPUT_DIR.resolve()}/")
    print(sep + "\n")


if __name__ == "__main__":
    main()
