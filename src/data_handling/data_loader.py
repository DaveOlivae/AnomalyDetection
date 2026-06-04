"""
Script containing functions to load the TEP dataset, and the TEPDatasetPaths dataclass 
which contains all the paths and parameters related to loading the TEP dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Iterable, Optional
from configs.tep_config import TEPDatasetPaths, TEPWindowConfig


logger = logging.getLogger(__name__)


def load_tep_csv(path: Path, label: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the TEP dataset from a CSV file, and optionally adds a label column.
    The label column is set to the provided label value for all rows if label is not None 
    """

    logger.info("Carregando dataset %s...", path)

    df = pd.read_csv(path)

    logger.info("Dataset carregado com sucesso! Shape: %s", df.shape)

    if label is not None:
        logger.info("Adicionando coluna de label com valor %s...", label)
        df["label"] = label
    return df


def choose_simulations(path: Path, n_sims: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """
    Chooses a specified number of simulation runs from the TEP dataset CSV file,
    using a random number generator for reproducibility. If n_sims is None, returns all simulations
    Returns a list of simulationRun IDs
    """

    # this reads only the simulationRun column, returns a dataframe tha is converted into a series by addressing
    # the column with ["simulationRun"], then we get the unique simulation runs with .unique()
    sim_ids = pd.read_csv(path, usecols=["simulationRun"])["simulationRun"].unique()

    # if n_sims is None, we return all simulation IDs
    if n_sims is None:
        logger.info("n_sims is None, returning all %d simulation IDs from %s.", len(sim_ids), path.name)
        return sim_ids

    # chooses a number of ids from sim_ids without replacement, this number is defined by the size parameter
    # if n_sims is greater than the number of available sim_ids, it will choose all sim_ids

    selected = rng.choice(sim_ids, size=min(n_sims, len(sim_ids)), replace=False)

    logger.info("Selected %d simulation IDs from %d available in %s.",
                len(selected),
                len(sim_ids),
                path.name
                )

    return selected


def load_selected_simulations(
    path: Path,
    label: int,
    sim_ids: Iterable[int],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Loads only the rows from the TEP dataset CSV file that correspond to the specified simulation runs.
    This is done by reading the CSV file in chunks, filtering each chunk for the selected simulation 
    """

    # turns the sim_ids iterable into a set for faster lookup
    selected_ids = set(sim_ids)
    chunks = []

    logger.info(
    "Loading %s (%d simulations)",
        path.name,
        len(selected_ids)
    )

    # reads the CSV file in chunks, each chunk is a dataframe with at most chunksize rows
    for chunk in pd.read_csv(path, chunksize=chunksize):

        # filters the chunk to keep only the rows where the simulationRun column is in the selected_ids set
        # the expression inside the brackets creates a boolean mask that is True for rows where simulationRun is in selected_ids, and False otherwise
        filtered = chunk[chunk["simulationRun"].isin(selected_ids)]

        if not filtered.empty:
            chunks.append(filtered.copy())

    if not chunks:
        logger.error(
            "No rows found in %s for selected simulations",
            path
        )

        raise ValueError(f"No rows found in {path} for selected simulations.")

    df = pd.concat(chunks, ignore_index=True)
    df["label"] = label

    logger.info(
        "Loaded %d rows from %s",
        len(df),
        path.name
    )

    return df


def split_by_simulation(
    df: pd.DataFrame,
    val_ratio: float,
    rng: np.random.Generator,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the TEP dataset dataframe into training and validation sets based on simulation runs,
    ensuring that all data from a given simulation run is in the same set. 
    The split is done separately for each class label to maintain an even class distribution in both sets. 
    """

    train_parts = []
    val_parts = []

    logger.info(
        "Splitting dataset into train and validation sets (val_ratio=%.2f)",
        val_ratio
    )

    # we're separating the dataframe by label to ensure that we have a
    # representative distribution of classes in both training and validation sets, 
    # especially if the dataset is imbalanced. By grouping by the label column, 
    # we can perform the split separately for each class, which helps maintain 
    # the class distribution in both sets.
    for label in sorted(df[label_col].unique()):

        subset = df[df[label_col] == label]

        sims = subset["simulationRun"].unique()

        rng.shuffle(sims)

        # the extra math steps are to deal with the possibility we have to little
        # simulations to split, in which case we want to put at least one simulation 
        # in the validation set, and if we have only one simulation,
        # then n_val will be set to 0, meaning that all simulations will go to the training set
        n_val = max(1, int(len(sims) * val_ratio)) if len(sims) > 1 else 0

        val_sims = sims[:n_val]
        train_sims = sims[n_val:]

        logger.info(
            "Label %s: %d simulations -> train=%d val=%d",
            label,
            len(sims),
            len(train_sims),
            len(val_sims)
        )

        train_parts.append(subset[subset["simulationRun"].isin(train_sims)])
        val_parts.append(subset[subset["simulationRun"].isin(val_sims)])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    logger.info(
        "Split complete. Train shape=%s | Validation shape=%s",
        train_df.shape,
        val_df.shape
    )

    return train_df, val_df


def infer_feature_columns(df: pd.DataFrame, meta_cols: Iterable[str]) -> list[str]:
    """
    Returns only the feature columns from the TEP dataset dataframe, excluding the meta columns and the label column. 
    """

    # we want to exclude the meta columns and the label column from the feature columns
    excluded = set(meta_cols) | {"label"}

    feature_columns = [
        column 
        for column in df.columns 
        if column not in excluded
    ]

    logger.info(
        "Identified %d feature columns",
        len(feature_columns)
    )

    return feature_columns


def load_binary_trainval_test(
    df_paths: TEPDatasetPaths = TEPDatasetPaths(),
    cfg: TEPWindowConfig = TEPWindowConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Loads the TEP dataset from the specified paths, samples the specified number of simulations for training and testing,
    splits the training data into training and validation sets, and returns the resulting dataframes along with
    the list of feature columns. The function uses the parameters defined in the TEPWindowConfig dataclass to control the loading and splitting process. 
    """

    logger.info(
        "Loading binary TEP dataset using random_state=%d",
        cfg.random_state
    )

    rng = np.random.default_rng(cfg.random_state)

    # train and validation sets
    ff_train_ids = choose_simulations(df_paths.ff_train, cfg.n_ff_train, rng)
    fa_train_ids = choose_simulations(df_paths.fa_train, cfg.n_fa_train, rng)

    ff_train = load_selected_simulations(df_paths.ff_train, 0, ff_train_ids, cfg.chunksize)
    fa_train = load_selected_simulations(df_paths.fa_train, 1, fa_train_ids, cfg.chunksize)

    trainval_df = pd.concat([ff_train, fa_train], ignore_index=True)

    logger.info(
        "Train/validation dataset shape=%s",
        trainval_df.shape
    )

    feature_columns = infer_feature_columns(trainval_df, cfg.meta_cols)

    logger.info(
        "Using %d feature columns",
        len(feature_columns)
    )

    train_df, val_df = split_by_simulation(trainval_df, cfg.val_ratio, rng)

    logger.info(
        "Train shape=%s | Validation shape=%s",
        train_df.shape,
        val_df.shape
    )

    # test set
    ff_test_ids = choose_simulations(df_paths.ff_test, cfg.n_ff_test, rng)
    fa_test_ids = choose_simulations(df_paths.fa_test, cfg.n_fa_test, rng)

    ff_test = load_selected_simulations(df_paths.ff_test, 0, ff_test_ids, cfg.chunksize)
    fa_test = load_selected_simulations(df_paths.fa_test, 1, fa_test_ids, cfg.chunksize)

    test_df = pd.concat([ff_test, fa_test], ignore_index=True)

    logger.info(
        "Test shape=%s",
        test_df.shape
    )

    logger.info(
        "Dataset loading pipeline completed successfully"
    )

    return train_df, val_df, test_df, feature_columns
