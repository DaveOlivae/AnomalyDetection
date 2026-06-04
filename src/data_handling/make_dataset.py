
"""
This is the main script for building the windowed datasets for the TEP anomaly detection project. 
It contains functions to load the TEP dataset, build windowed features and labels for training, 
validation, and testing splits, and save the resulting datasets to compressed .npz files along 
with metadata information. The script uses the configurations defined in TEPDatasetPaths and 
TEPWindowConfig dataclasses to specify the paths to the dataset CSV files and the parameters 
for building the windows. The main functions in this script are build_binary_window_splits, 
which creates the windowed datasets for each split, and save_window_splits, which saves the 
datasets and metadata to disk for future use in modeling and evaluation steps of the pipeline.
"""

import logging
import numpy as np
from pathlib import Path
from dataclasses import asdict
from configs.tep_config import TEPDatasetPaths, TEPWindowConfig
from .data_loader import load_binary_trainval_test
from .create_windows import make_feature_names, build_windows
from src.modeling.persistence import save_json

logger = logging.getLogger(__name__)

def build_binary_window_splits(
    df_paths: TEPDatasetPaths = TEPDatasetPaths(),
    cfg: TEPWindowConfig = TEPWindowConfig(),
) -> dict[str, object]:
    """
    Loads the TEP dataset, builds windowed features and labels for training, validation, and testing sets, and returns them in a dictionary along with feature names and config parameters. 
    The function uses the load_binary_trainval_test function to load and split the dataset, and the build_windows function to create the windowed datasets for each split. 
    The resulting dictionary contains the windowed features and labels for each split, as well as the feature names and the configuration used for building the windows.
    The returned dictionary has the following structure:
     {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "feature_columns": feature_columns,
        "feature_names": feature_names,
        "config": config_dict
     }
    where X_train, X_val, X_test are 2D numpy arrays of shape (n_samples, n_features) containing the windowed features for each split,
    y_train, y_val, y_test are 1D numpy arrays of shape (n_samples,) containing the corresponding binary labels for each split,
    feature_columns is a list of the original feature column names from the TEP dataset,
    feature_names is a list of the generated feature names for the windowed dataset based on the mode and window size,
    and config_dict is a dictionary containing the parameters from the TEPWindowConfig dataclass used for building the windows. 
    """

    logger.info("Starting binary window dataset creation")

    train_df, val_df, test_df, feature_columns = load_binary_trainval_test(df_paths, cfg)

    logger.info(
    "Datasets loaded successfully | train=%s | val=%s | test=%s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )

    logger.info(
        "Using %d feature columns",
        len(feature_columns)
    )

    splits = {}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(
            "Building windows for %s split",
            split_name
        )

        splits[split_name] = build_windows(
            split_df,
            feature_columns,
            window_size=cfg.window_size,
            stride=cfg.stride,
            mode=cfg.mode,
        )

        X, y = splits[split_name]

        logger.info(
            "%s split generated: X=%s | y=%s",
            split_name,
            X.shape,
            y.shape,
        )

    logger.info("Window dataset creation completed")

    return {
        **splits,
        "feature_columns": feature_columns,
        "feature_names": make_feature_names(feature_columns, cfg.mode, cfg.window_size),
        "config": asdict(cfg),
    }


def save_window_splits(dataset: dict[str, object], output_dir: Path) -> None:
    """
    Saves the windowed datasets for training, validation, and testing splits to compressed 
    .npz files in the specified output directory, along with a metadata.json file containing 
    feature names and configuration parameters. The function iterates over the splits in the 
    dataset dictionary, saves each split's features and labels to a separate .npz file, and 
    then saves the metadata information to a JSON file for future reference. The .npz files 
    are saved with names "train.npz", "val.npz", and "test.npz" corresponding to each split.
    The metadata.json file contains the original feature columns from the TEP dataset, the 
    generated feature names for the windowed dataset, and the configuration parameters used 
    for building the windows. This allows for easy loading and understanding of the windowed 
    datasets in future steps of the modeling pipeline.
    """

    logger.info(
        "Saving window datasets to %s",
        output_dir
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = np.asarray(dataset["feature_names"], dtype=str)

    for split in ["train", "val", "test"]:

        X, y = dataset[split]

        output_file = output_dir / f"{split}.npz"

        logger.info(
            "Saving %s split | X=%s | y=%s",
            split,
            X.shape,
            y.shape,
        )

        np.savez_compressed(
            output_file, 
            X=X, 
            y=y, 
            feature_names=feature_names
        )

        logger.info(
            "Saved %s",
            output_file.name
        )

    logger.info("Saving metadata.json")

    save_json(
        {
            "feature_columns": dataset["feature_columns"],
            "feature_names": dataset["feature_names"],
            "config": dataset["config"],
        },
        output_dir / "metadata.json",
    )

    logger.info(
        "All window datasets saved successfully"
    )


def load_window_split(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads a windowed dataset split from a compressed .npz file, returning the features, labels, 
    and feature names. The function reads the .npz file using numpy's load function, which 
    returns a dictionary-like object containing the arrays stored in the file. It extracts the 
    features (X), labels (y), and feature names from the loaded data, ensuring that the feature 
    names are returned as a list of strings for consistency with the rest of the pipeline. 
    This function allows for easy loading of the windowed datasets that were previously saved 
    using the save_window_splits function, enabling seamless integration into the modeling and 
    evaluation steps of the anomaly detection pipeline.
    """

    logger.info(
        "Loading window dataset from %s",
        path
    )

    data = np.load(path, allow_pickle=False)

    logger.info(
        "Loaded split successfully | X=%s | y=%s",
        data["X"].shape,
        data["y"].shape,
    )

    return data["X"], data["y"], data["feature_names"].astype(str).tolist()


# Bloco de execução principal
if __name__ == "__main__":
    print("Iniciando o processamento do Dataset TEP...")

    from configs.logger import setup_logger

    logger = setup_logger(Path("dataset_creation.log"))
    
    # 1. Configurações padrão
    caminhos = TEPDatasetPaths()
    configuracao = TEPWindowConfig()
    
    # 2. Processa tudo e gera as janelas
    dataset_final = build_binary_window_splits(caminhos, configuracao)
    print(dataset_final)
    
    # 3. Salva os resultados compactados na pasta escolhida (ex: 'data/processed')
    pasta_saida = Path("./data/final")
    save_window_splits(dataset_final, pasta_saida)
    
    print(f"Sucesso! Arquivos salvos com segurança na pasta: {pasta_saida}")
