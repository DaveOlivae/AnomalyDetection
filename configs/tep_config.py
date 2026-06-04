"""
Configuration dataclasses for loading and processing the TEP dataset.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from . import paths
from . import settings


TEP_META_COLS = ["faultNumber", "simulationRun", "sample"]


# Define a type for the windowing mode, 
# which can be either "stats" or "flatten"
WindowMode = Literal["stats", "flatten"]


@dataclass
class TEPDatasetPaths:
    """
    Dataclass containing all the paths related 
    to loading the TEP dataset. 
    """

    # the paths are stored in the configs/paths.py file,
    # but we can set defaults here for convenience
    # if we want, we can change the paths when creating 
    # an instance of TEPDatasetPaths
    ff_train: Path = paths.PROCESSED_FF_TRAIN
    fa_train: Path = paths.PROCESSED_FA_TRAIN
    ff_test: Path = paths.PROCESSED_FF_TEST
    fa_test: Path = paths.PROCESSED_FA_TEST


@dataclass
class TEPWindowConfig:
    """
    Dataclass containing all the parameters related to
    building windowed datasets from the TEP dataset. 
    """

    window_size: int = 20
    stride: int = 5
    mode: WindowMode = "stats"
    n_ff_train: int = 120
    n_fa_train: int = 6
    n_ff_test: int = 60
    n_fa_test: int = 3
    chunksize: int = 200_000
    val_ratio: float = 0.2
    random_state: int = settings.RANDOM_STATE
    # copies of the TEP_META_COLS list to avoid mutable 
    # default argument issues
    meta_cols: list[str] = field(default_factory=lambda: list(TEP_META_COLS))
