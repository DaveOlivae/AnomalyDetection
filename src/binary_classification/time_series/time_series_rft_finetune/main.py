import numpy as np
import pandas as pd
from typing import List


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
):
    
    grp = group.sort_values("sample")
    vals = grp[feat_cols].values.astype(np.float32)
    labels = grp["label"].values
    T = len(vals)

    X_list, y_list = [], []

    for start in range(0, T - window + 1, stride):
        w = vals[start : start + window]
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
        desc: str
):
    """
    Parameters:
    df: the complete dataframe
    feat_cols: list with the names of columns that are just features
    window: window size
    stride: how much we shift at every window
    desc: description for printing
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