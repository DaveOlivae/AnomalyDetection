"""
Change variable names to correct names on the TEP dataset
"""

import json
from pathlib import Path
import logging
import pandas as pd
from configs.paths import (
    VARIABLE_NAMES,
    FAULT_FREE_TRAIN,
    FAULTY_TRAIN,
    FAULT_FREE_TEST,
    FAULTY_TEST,
    PROCESSED_FF_TRAIN,
    PROCESSED_FA_TRAIN,
    PROCESSED_FF_TEST,
    PROCESSED_FA_TEST,
)

logger = logging.getLogger(__name__)


def _load_variable_names():
    """
    Load the json file with the variable names, and returns a dict from it
    """

    logger.info(f"Carregando os nomes das variáveis do arquivo {VARIABLE_NAMES}...")

    with open(VARIABLE_NAMES, "r", encoding='utf-8') as f:
        var_names = json.load(f)

    logger.info(f"Nomes das variáveis carregados com sucesso! {len(var_names)} variáveis encontradas.")

    return var_names


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the standard column names on the dataset for the actual
    variable names
    """

    mapping = _load_variable_names()
    mapping_upper = {k.upper(): v for k, v in mapping.items()}

    new_columns = {
        col: mapping_upper[col.upper()]
        for col in df.columns
        if col.upper() in mapping_upper
    }

    logger.info(f"Renomeando as colunas: {df.columns.tolist()} -> {list(new_columns.values())}")

    return df.rename(columns=new_columns)


def preprocess(input_path: Path, output_path:Path, logger: logging.Logger = None):
    """
    Loads the data, rename columns and saves the new data 
    """

    if output_path.exists():
        logger.info(f"Arquivo {output_path} já existe, pulando processamento...")
        return

    logger.info(f"Carregando dataset {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Dataset carregado com sucesso! Shape: {df.shape}")

    # renomeia as variaveis do dataset de treino
    logger.info("Renomeando as colunas...")
    df = rename_columns(df)

    # salva como csv
    logger.info("Salvando os dados processados para o arquivo {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Dados processados e salvos com sucesso!")


# TODO: mudar o nome dessa função, talvez para preprocess_all() ou algo do tipo?
def preprocess_tep():
    """
    The function will load all the files, correct them and then save them
    """

    logger.info("Iniciando o pré-processamento dos dados do TEP...")

    in_out_paths = [(FAULT_FREE_TRAIN, PROCESSED_FF_TRAIN),
                    (FAULTY_TRAIN, PROCESSED_FA_TRAIN),
                    (FAULT_FREE_TEST, PROCESSED_FF_TEST),
                    (FAULTY_TEST, PROCESSED_FA_TEST)]

    for file in in_out_paths:
        preprocess(file[0], file[1], logger)

    logger.info("Pré-processamento dos dados do TEP concluído com sucesso!")


if __name__ == "__main__":
    from configs.logger import setup_logger

    logger = setup_logger(Path("preprocess.log"))

    preprocess_tep()
