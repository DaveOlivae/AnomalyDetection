"""
This module contains functions for creating windowed features and labels from the TEP dataset dataframe.
The main function is build_binary_window_splits, which loads the TEP dataset, builds windowed features and 
labels for training, validation, and testing sets, and returns them in a dictionary along with feature names 
and config parameters. 
The function uses the load_binary_trainval_test function to load and split the dataset, and the build_windows 
function to create the windowed datasets for each split. 
The resulting dictionary contains the windowed features and labels for each split, as well as the feature 
names and the configuration used for building the windows.
The returned dictionary has the following structure:
 {
    "train": (X_train, y_train),
    "val": (X_val, y_val),
    "test": (X_test, y_test),
    "feature_columns": feature_columns,
    "feature_names": feature_names,
    "config": config_dict
 }
where X_train, X_val, X_test are 2D numpy arrays of shape (n_samples, n_features) containing the windowed 
features for each split,
y_train, y_val, y_test are 1D numpy arrays of shape (n_samples,) containing the corresponding binary labels 
for each split,
feature_columns is a list of the original feature column names from the TEP dataset,
feature_names is a list of the generated feature names for the windowed dataset based on the mode and 
window size,
and config_dict is a dictionary containing the parameters from the TEPWindowConfig dataclass used for 
building the windows. 
The module also contains a helper function save_window_splits, which saves the windowed datasets for 
training, validation, and testing splits to compressed numpy files in a specified output directory.
"""

import logging
import numpy as np
import pandas as pd
from configs.tep_config import WindowMode

logger = logging.getLogger(__name__)

def make_feature_names(feature_columns: list[str], mode: WindowMode, window_size: int) -> list[str]:
    """
    Generates feature names for the windowed dataset based on the original feature columns and the windowing mode.
    If mode is "stats", the feature names will be in the format of "{stat}_{column}", 
    where stat is one of "mean", "std", "min", "max". 
    If mode is "flatten", the feature names will be in the format of "t{step}_{column}",
    where step is the time step within the window (0 to window_size-1).
    """

    if mode == "stats":
        return [f"{stat}_{column}" for stat in ["mean", "std", "min", "max"] for column in feature_columns]

    return [f"t{step}_{column}" for step in range(window_size) for column in feature_columns]


def extract_window_features(values: np.ndarray, mode: WindowMode) -> np.ndarray:
    """
    Extracts features from a window of values based on the specified mode.
    If mode is "stats", it computes the mean, standard deviation, minimum, and maximum for each 
    feature column across the window, and concatenates them into a single feature vector.
    If mode is "flatten", it simply flattens the window of values into a single feature vector by 
    """

    if mode == "flatten":
        return values.reshape(-1)
    return np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
        ]
    )


def windows_from_group(
    group: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    stride: int,
    mode: WindowMode,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds windowed features and labels from a group of rows corresponding to a 
    single simulation run and fault number.
    The function iterates over the rows of the 
    group with a sliding window approach, 
    extracting features 
    """

    # since its a time series the dataframe must be in order of samples
    ordered = group.sort_values("sample")

    # here we isolate the feature and labels into numpy arrays 
    # for faster processing, we also convert the feature values 
    # to float32 to save memory
    values = ordered[feature_columns].to_numpy(dtype=np.float32)
    labels = ordered["label"].to_numpy()

    # we calculate the number of features in the output 
    # windowed dataset based on the mode and the number of feature columns
    n_features = len(make_feature_names(feature_columns, mode, window_size))

    X_rows = []
    y_rows = []

    # here is where we actually build the windows, we iterate over the values array with a sliding window approach,
    # the start index of the window goes from 0 to the length of the values array
    for start in range(0, len(values) - window_size + 1, stride):

        end = start + window_size
        
        X_rows.append(extract_window_features(values[start:end], mode))
        y_rows.append(int(labels[start:end].mean() >= 0.5))

    if not X_rows:
        return np.empty((0, n_features), dtype=np.float32), np.empty(0, dtype=np.int8)

    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int8)


def build_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 20,
    stride: int = 5,
    mode: WindowMode = "stats",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds windowed features and labels from the TEP dataset dataframe by grouping the data by 
    simulation run and fault number, 
    and applying the windows_from_group function to each group.
    The resulting windowed features and labels from all groups are concatenated into single arrays for the entire dataset.
    The function returns a tuple of (X, y), where X is a 2D array of shape (n_samples, n_features) containing the windowed features,
    and y is a 1D array of shape (n_samples,) containing the corresponding binary labels.
    """

    logger.info(
        "Building windows: mode=%s | window_size=%d | stride=%d",
        mode,
        window_size,
        stride,
    )

    logger.info(
        "Input dataframe shape=%s",
        df.shape
    )

    X_parts = []
    y_parts = []

    groups = df.groupby(
        ["simulationRun", "faultNumber"], 
        sort=False
    )

    n_groups = df.groupby(
        ["simulationRun", "faultNumber"],
        sort=False
    ).ngroups

    logger.info(
        "Found %d simulation groups",
        n_groups
    )

    # create groups based on simulationRun and faultNumber 
    for _, group in groups:

        # passes the group to the windows_from_group function, which returns the windowed features and labels for that group,
        X_group, y_group = windows_from_group(group, feature_columns, window_size, stride, mode)

        if len(X_group):
            X_parts.append(X_group)
            y_parts.append(y_group)

    if not X_parts:
        logger.warning(
            "No windows were generated."
        )

        n_features = len(make_feature_names(feature_columns, mode, window_size))
        return np.empty((0, n_features), dtype=np.float32), np.empty(0, dtype=np.int8)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts).astype(np.int8)

    logger.info(
        "Window dataset shape=%s",
        X.shape
    )

    logger.info(
        "Generated %d windows with %d features",
        X.shape[0],
        X.shape[1],
    )

    logger.info(
        "Class distribution: normal=%d anomalous=%d",
        np.sum(y == 0),
        np.sum(y == 1),
    )

    return X, y

