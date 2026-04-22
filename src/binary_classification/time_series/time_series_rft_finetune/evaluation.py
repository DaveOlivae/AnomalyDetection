"""
evaluation.py
═════════════
Funções reutilizáveis de avaliação de modelos para o projeto TEP.

Conteúdo:
  compute_metrics()          → dicionário com todas as métricas
  print_report()             → classification report no terminal
  plot_confusion_matrix()    → matriz de confusão de um único modelo
  plot_confusion_matrices()  → grade com vários modelos lado a lado
  plot_metrics_comparison()  → gráfico de barras comparando modelos
  plot_feature_importance()  → importância das features (RF / LGBM)
  plot_learning_curve()      → curva de validação do RandomizedSearch
  plot_pr_curve()            → curva Precision-Recall
  plot_roc_curve()           → curva ROC
  save_metrics_csv()         → salva tabela de métricas em CSV
  save_model()               → serializa modelo com joblib
  load_model()               → carrega modelo serializado
  full_evaluation()          → avalia modelo e gera todos os gráficos
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ── Paleta & estilo globais ────────────────────────────────────────────────────
_PALETTE = {
    "normal": "#4C72B0",
    "fault":  "#DD8452",
    "good":   "#55A868",
    "bad":    "#C44E52",
}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


# ══════════════════════════════════════════════════════════════════════════════
#  Métricas
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calcula as métricas clássicas de classificação binária.

    Parâmetros
    ----------
    y_true : rótulos verdadeiros
    y_pred : rótulos preditos (0 ou 1)
    y_prob : probabilidades da classe positiva (opcional) — habilita AUC e AP

    Retorna
    -------
    Dicionário com: Acurácia, Balanced Accuracy, Precisão, Recall,
                    F1, ROC-AUC, Avg Precision
    """
    metrics: Dict[str, float] = {
        "Acurácia":          accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precisão":          precision_score(y_true, y_pred, zero_division=0),
        "Recall":            recall_score(y_true, y_pred, zero_division=0),
        "F1":                f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"]       = roc_auc_score(y_true, y_prob)
        metrics["Avg Precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["ROC-AUC"]       = float("nan")
        metrics["Avg Precision"] = float("nan")
    return metrics


def print_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Imprime classification report formatado no terminal."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Modelo: {model_name}")
    print(sep)
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal (0)", "Falha  (1)"],
        digits=4,
    ))
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:<22}: {v:.4f}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  Matrizes de Confusão
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plota a matriz de confusão de um único modelo.

    Se `ax` for passado, desenha nele (útil para grade de subplots).
    Se `out_path` for passado, salva o arquivo.
    """
    cm    = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Falha"],
        yticklabels=["Normal", "Falha"],
        ax=ax, linewidths=0.5, cbar=own_fig,
    )
    # Percentuais em cinza abaixo de cada contagem
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.72,
                f"({cm[i, j] / total * 100:.1f}%)",
                ha="center", va="center", fontsize=8, color="grey",
            )

    ax.set_title(model_name, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Predito", fontsize=9)
    ax.set_ylabel("Real",    fontsize=9)

    if own_fig:
        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, bbox_inches="tight")
            print(f"  Salvo: {out_path}")
        plt.close(fig)

    return fig


def plot_confusion_matrices(
    cms_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Grade de matrizes de confusão para múltiplos modelos.

    Parâmetros
    ----------
    cms_dict : {nome_modelo: (y_true, y_pred)}
    out_path : caminho para salvar (opcional)
    """
    n   = len(cms_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, (y_true, y_pred)) in zip(axes, cms_dict.items()):
        plot_confusion_matrix(y_true, y_pred, name, ax=ax)

    fig.suptitle(
        "Matrizes de Confusão — TEP Binary Classification",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Comparação entre modelos
# ══════════════════════════════════════════════════════════════════════════════

def plot_metrics_comparison(
    results_df: pd.DataFrame,
    out_path: Optional[Path] = None,
    metrics_cols: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Gráfico de barras agrupadas comparando modelos em várias métricas.

    Parâmetros
    ----------
    results_df   : DataFrame com modelos como índice e métricas como colunas
    out_path     : caminho para salvar (opcional)
    metrics_cols : colunas a plotar (padrão: Acurácia, Precisão, Recall, F1, ROC-AUC)
    """
    if metrics_cols is None:
        metrics_cols = ["Acurácia", "Precisão", "Recall", "F1", "ROC-AUC"]

    cols = [c for c in metrics_cols if c in results_df.columns]
    data = results_df[cols].astype(float)
    n_models, n_metrics = data.shape

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 5))
    x      = np.arange(n_models)
    w      = 0.8 / n_metrics
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_metrics))

    for i, (col, color) in enumerate(zip(cols, colors)):
        offset = (i - n_metrics / 2 + 0.5) * w
        bars   = ax.bar(x + offset, data[col], w, label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — TEP", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    model,
    feat_names: List[str],
    model_name: str,
    out_path: Optional[Path] = None,
    top_n: int = 25,
) -> Optional[plt.Figure]:
    """
    Plota as top_n features mais importantes.
    Suporta modelos com `.feature_importances_` (RF, LGBM, XGB, etc.).
    Retorna None silenciosamente se o modelo não tiver o atributo.
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None

    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1.5))
    ax.barh(np.array(feat_names)[idx], imp[idx], color=_PALETTE["normal"])
    ax.set_xlabel("Importância (Impureza / Ganho)")
    ax.set_title(f"Top {top_n} Features — {model_name}", fontweight="bold")
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Curvas ROC e Precision-Recall
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve(
    models_probs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Curva ROC para um ou mais modelos.

    Parâmetros
    ----------
    models_probs : {nome: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.8, len(models_probs)))

    for (name, (y_true, y_prob)), color in zip(models_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val     = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR (1 - Especificidade)")
    ax.set_ylabel("TPR (Recall)")
    ax.set_title("Curva ROC — TEP", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


def plot_pr_curve(
    models_probs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Curva Precision-Recall para um ou mais modelos.

    Parâmetros
    ----------
    models_probs : {nome: (y_true, y_prob)}
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.8, len(models_probs)))

    for (name, (y_true, y_prob)), color in zip(models_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap           = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_title("Curva Precision-Recall — TEP", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Curva de aprendizado do RandomizedSearch
# ══════════════════════════════════════════════════════════════════════════════

def plot_search_results(
    cv_results: dict,
    param_name: str,
    score_name: str = "mean_test_score",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plota score médio de validação cruzada vs. um hiperparâmetro.
    Útil para visualizar o resultado do RandomizedSearchCV.

    Parâmetros
    ----------
    cv_results : `search.cv_results_` do sklearn
    param_name : nome do parâmetro (ex: "param_n_estimators")
    score_name : coluna de score a plotar
    """
    df = pd.DataFrame(cv_results)
    if param_name not in df.columns:
        print(f"  [AVISO] Parâmetro '{param_name}' não encontrado em cv_results.")
        return None

    df = df.sort_values(param_name)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[param_name], df[score_name], "o-", color=_PALETTE["normal"], ms=5)
    if "std_test_score" in df.columns:
        ax.fill_between(
            df[param_name],
            df[score_name] - df["std_test_score"],
            df[score_name] + df["std_test_score"],
            alpha=0.2, color=_PALETTE["normal"],
        )
    ax.set_xlabel(param_name.replace("param_", ""))
    ax.set_ylabel(score_name.replace("_", " ").title())
    ax.set_title("Resultado da Busca de Hiperparâmetros", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Salvo: {out_path}")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Persistência
# ══════════════════════════════════════════════════════════════════════════════

def save_metrics_csv(
    results: Dict[str, Dict[str, float]],
    out_path: Path,
    sort_by: str = "F1",
) -> pd.DataFrame:
    """
    Salva tabela de métricas em CSV e retorna o DataFrame ordenado.

    Parâmetros
    ----------
    results : {nome_modelo: {métrica: valor}}
    """
    df = pd.DataFrame(results).T.round(4)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    df.to_csv(out_path)
    print(f"  Salvo: {out_path}")
    return df


def save_model(model, path: Path) -> None:
    """Serializa modelo com joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Modelo salvo: {path}")


def load_model(path: Path):
    """Carrega modelo serializado com joblib."""
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Avaliação completa (all-in-one)
# ══════════════════════════════════════════════════════════════════════════════

def full_evaluation(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Avalia um modelo e gera todos os gráficos relevantes.

    Gera (se out_dir for fornecido):
      - confusion_matrix_{model_name}.png
      - feature_importance_{model_name}.png  (se aplicável)
      - roc_curve_{model_name}.png           (se predict_proba disponível)
      - pr_curve_{model_name}.png            (se predict_proba disponível)

    Retorna dicionário de métricas.
    """
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    safe = model_name.replace(" ", "_").lower()
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_report(model_name, y_test, y_pred, metrics)

    if out_dir:
        plot_confusion_matrix(
            y_test, y_pred, model_name,
            out_path=out_dir / f"confusion_matrix_{safe}.png",
        )
        if feat_names:
            plot_feature_importance(
                model, feat_names, model_name,
                out_path=out_dir / f"feature_importance_{safe}.png",
            )
        if y_prob is not None:
            plot_roc_curve(
                {model_name: (y_test, y_prob)},
                out_path=out_dir / f"roc_curve_{safe}.png",
            )
            plot_pr_curve(
                {model_name: (y_test, y_prob)},
                out_path=out_dir / f"pr_curve_{safe}.png",
            )

    return metrics
