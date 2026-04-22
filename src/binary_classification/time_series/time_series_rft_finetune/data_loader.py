"""
data_loader.py
══════════════
Responsável por:
  1. Carregar os 4 CSVs do TEP com controle de memória (amostragem por simulação)
  2. Garantir balanceamento 50/50 (Normal vs Falha) sem quebrar séries temporais
  3. Gerar features via sliding window (estatísticas por janela)
  4. Dividir em train / val / test respeitando fronteiras de simulação

Uso típico:
    from data_loader import TEPLoader, LoaderConfig

    cfg    = LoaderConfig()
    loader = TEPLoader(cfg)
    data   = loader.load_all()

    X_train, y_train = data["train"]
    X_val,   y_val   = data["val"]
    X_test,  y_test  = data["test"]
    feat_names       = data["feat_names"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  Configuração
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LoaderConfig:
    """
    Config class for the data loading of the TEP dataset.

    ======== Path config ==================
    We define the overall data path as 'data_dir'
    And we define the 4 paths for the train and test faulty and fault free files as:
    'ff_train_file', 'fa_train_file', 'ff_test_file' and 'fa_test_file'

    ======== Memory Control ===============
    Lembrete da estrutura do TEP:
        Treino  → ff: 500 sims x 500  rows  |  fa: 500 sims x 20 falhas x 500  rows
        Teste   → ff: 500 sims x 960  rows  |  fa: 500 sims x 20 falhas x 960  rows
    
      Para 50/50 precisamos de ~20x mais sims fault-free que faulty
      (pois cada sim faulty já carrega 20 falhas x 500 rows = 10 000 rows,
       enquanto cada sim ff tem apenas 500 rows).
    
      Valores padrão ➜ memória estimada < 2 GB:
        Treino  → 120 ff x 500  = 60 000  rows  |  6 fa x 10 000  = 60 000  rows
        Teste   →  60 ff x 960  = 57 600  rows  |  3 fa x 19 200  = 57 600  rows

    ======== Project Parameters ============
    Here we define a coupe of parameters that are project specific, like the ratio of
    train/val, the window size and stride (since we're dealing with time series), the
    random state and the metadata of the dataset (not used as features)
    """

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("../data/raw")

    ff_train_file: str = "TEP_FaultFree_Training.csv"
    fa_train_file: str = "TEP_Faulty_Training.csv"
    ff_test_file:  str = "TEP_FaultFree_Testing.csv"
    fa_test_file:  str = "TEP_Faulty_Testing.csv"

    # ── Orçamento de simulações (controle de memória) ─────────────────────────
    
    n_ff_train: int = 120   # sims fault-free para treino+val
    n_fa_train: int = 6     # sims faulty    para treino+val
    n_ff_test:  int = 60    # sims fault-free para teste
    n_fa_test:  int = 3     # sims faulty    para teste

    # ── Split train / val (proporção sobre as sims de treino) ─────────────────
    val_ratio: float = 0.20  # 20 % das simulações vão para validação

    # ── Sliding window ────────────────────────────────────────────────────────
    window_size: int = 20
    stride:      int = 5     # passo entre janelas (overlap de 75 %)

    # ── Reprodutibilidade ─────────────────────────────────────────────────────
    random_state: int = 42

    # ── Colunas de metadados (não usadas como features) ───────────────────────
    meta_cols: List[str] = field(
        default_factory=lambda: ["faultNumber", "simulationRun", "sample"]
    )

    @property
    def ff_train_path(self) -> Path:
        return self.data_dir / self.ff_train_file

    @property
    def fa_train_path(self) -> Path:
        return self.data_dir / self.fa_train_file

    @property
    def ff_test_path(self) -> Path:
        return self.data_dir / self.ff_test_file

    @property
    def fa_test_path(self) -> Path:
        return self.data_dir / self.fa_test_file


# ══════════════════════════════════════════════════════════════════════════════
#  TEPLoader
# ══════════════════════════════════════════════════════════════════════════════
class TEPLoader:
    def __init__(self, config: LoaderConfig | None = None):
        self.cfg = config or LoaderConfig()
        self.rng = np.random.default_rng(self.cfg.random_state)

    # ── Utilitários internos ──────────────────────────────────────────────────

    def _sample_sims(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Amostra n simulações completas (por simulationRun) sem quebrar séries."""
        all_sims = df["simulationRun"].unique()
        chosen   = self.rng.choice(all_sims, size=min(n, len(all_sims)), replace=False)
        return df[df["simulationRun"].isin(chosen)].copy()

    @staticmethod
    def _load_csv(path: Path, label: int, tag: str) -> pd.DataFrame:
        print(f"  [{tag}] Carregando {path.name} ...", end="", flush=True)
        df          = pd.read_csv(path)
        df["label"] = label
        print(f" {len(df):>10,} linhas | {df['simulationRun'].nunique()} sims")
        return df

    @staticmethod
    def _infer_feat_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
        return [c for c in df.columns if c not in meta_cols + ["label"]]

    # ── Sliding window ────────────────────────────────────────────────────────

    @staticmethod
    def _window_features(w: np.ndarray) -> np.ndarray:
        """
        Features extraídas de uma janela (window_size × n_vars):
          média, desvio padrão, mínimo, máximo por variável → 4 × n_vars valores
        """
        return np.concatenate([
            w.mean(axis=0),
            w.std(axis=0),
            w.min(axis=0),
            w.max(axis=0),
        ])

    def _windows_from_series(
        self,
        series: pd.DataFrame,
        feat_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica sliding window a uma única série temporal (uma sim × falha).
        Label da janela = maioria dos rótulos no intervalo (trivial: todos iguais).
        """
        s      = series.sort_values("sample")
        vals   = s[feat_cols].values.astype(np.float32)
        labels = s["label"].values
        T      = len(vals)
        ws, st = self.cfg.window_size, self.cfg.stride
        n_feat = 4 * len(feat_cols)

        Xs, ys = [], []
        for start in range(0, T - ws + 1, st):
            w   = vals[start : start + ws]
            lbl = labels[start : start + ws]
            Xs.append(self._window_features(w))
            ys.append(int(lbl.mean() >= 0.5))

        if not Xs:
            return np.empty((0, n_feat), dtype=np.float32), np.empty(0, dtype=np.int8)

        return (
            np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.int8),
        )

    def _build_windows(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        desc: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Itera sobre cada série temporal (sim × faultNumber) e empilha as janelas."""
        print(f"  Gerando janelas [{desc}]"
              f" (window={self.cfg.window_size}, stride={self.cfg.stride}) ...",
              end="", flush=True)

        Xs, ys = [], []
        for _, grp in df.groupby(["simulationRun", "faultNumber"], sort=False):
            x, y = self._windows_from_series(grp, feat_cols)
            if len(x):
                Xs.append(x)
                ys.append(y)

        X = np.vstack(Xs)
        y = np.concatenate(ys).astype(np.int8)
        counts = np.bincount(y.astype(int), minlength=2)
        print(f" {X.shape[0]:,} janelas | Normal={counts[0]:,} | Falha={counts[1]:,}")
        return X, y

    # ── Split train / val por simulação ───────────────────────────────────────

    def _split_by_sim(self, 
                      df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide o DataFrame de treino em train e val respeitando fronteiras
        de simulação (nunca quebra uma série temporal).

        A divisão é feita separadamente para fault-free e faulty para
        garantir que o balanceamento 50/50 se mantenha nos dois subsets.
        """

        val_ratio = self.cfg.val_ratio
        rng = self.rng

        train_parts, val_parts = [], []

        for label_val in df["label"].unique():
            sub  = df[df["label"] == label_val]
            sims = sub["simulationRun"].unique()
            rng.shuffle(sims)
            n_val = max(1, int(len(sims) * val_ratio))
            val_sims = sims[:n_val]
            train_sims = sims[n_val:]
            train_parts.append(sub[sub["simulationRun"].isin(train_sims)])
            val_parts.append(sub[sub["simulationRun"].isin(val_sims)])

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts,   ignore_index=True)

        _summary = lambda d: (
            d["simulationRun"].nunique(),
            dict(d["label"].value_counts().sort_index()),
        )
        tr_sims, tr_dist = _summary(train_df)
        va_sims, va_dist = _summary(val_df)
        print(f"  Split train/val → "
              f"Train: {tr_sims} sims {tr_dist} | "
              f"Val: {va_sims} sims {va_dist}")

        return train_df, val_df

    # ── API pública ───────────────────────────────────────────────────────────

    def load_all(self) -> Dict:
        """
        Ponto de entrada principal.

        Retorna dicionário com:
          "train"      → (X_train, y_train)
          "val"        → (X_val,   y_val)
          "test"       → (X_test,  y_test)
          "feat_names" → lista de nomes das features
          "feat_cols"  → nomes das colunas originais do processo
        """

        sep = "═" * 60
        cfg = self.cfg

        # ── Treino + Val ──────────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  CARREGANDO DADOS DE TREINO")
        print(sep)

        ff_tr = self._load_csv(cfg.ff_train_path, label=0, tag="FF-TRAIN")
        fa_tr = self._load_csv(cfg.fa_train_path, label=1, tag="FA-TRAIN")
        fa_tr["label"] = 1  # substitui todos os tipos de falha por 1

        ff_sub = self._sample_sims(ff_tr, cfg.n_ff_train)
        fa_sub = self._sample_sims(fa_tr, cfg.n_fa_train)
        del ff_tr, fa_tr

        trainval_df = pd.concat([ff_sub, fa_sub], ignore_index=True)
        del ff_sub, fa_sub

        feat_cols  = self._infer_feat_cols(trainval_df, cfg.meta_cols)
        feat_names = [
            f"{stat}_{col}"
            for stat in ["mean", "std", "min", "max"]
            for col in feat_cols
        ]

        train_df, val_df = self._split_by_sim(trainval_df)
        del trainval_df

        X_train, y_train = self._build_windows(train_df, feat_cols, "TRAIN")
        X_val,   y_val   = self._build_windows(val_df,   feat_cols, "VAL")
        del train_df, val_df

        # ── Teste ─────────────────────────────────────────────────────────────
        print(f"\n{sep}")
        print("  CARREGANDO DADOS DE TESTE")
        print(sep)

        ff_te = self._load_csv(cfg.ff_test_path, label=0, tag="FF-TEST")
        fa_te = self._load_csv(cfg.fa_test_path, label=1, tag="FA-TEST")
        fa_te["label"] = 1

        ff_sub = self._sample_sims(ff_te, cfg.n_ff_test)
        fa_sub = self._sample_sims(fa_te, cfg.n_fa_test)
        del ff_te, fa_te

        test_df = pd.concat([ff_sub, fa_sub], ignore_index=True)
        del ff_sub, fa_sub

        X_test, y_test = self._build_windows(test_df, feat_cols, "TEST")
        del test_df

        print(f"\n{sep}")
        print("  RESUMO DOS SPLITS")
        print(sep)
        for name, X, y in [
            ("Train", X_train, y_train),
            ("Val",   X_val,   y_val),
            ("Test",  X_test,  y_test),
        ]:
            c = np.bincount(y.astype(int), minlength=2)
            print(f"  {name:<6}: {X.shape[0]:>7,} janelas | "
                  f"Normal={c[0]:,} | Falha={c[1]:,} | Shape={X.shape}")

        return {
            "train":      (X_train, y_train),
            "val":        (X_val,   y_val),
            "test":       (X_test,  y_test),
            "feat_names": feat_names,
            "feat_cols":  feat_cols,
        }
