import pandas as pd
import json
import sys


RAW_PATH = "/home/davideoliveira/Programming/AIProjects/AnomalyDetection/data/raw/"
PROCESSED_PATH = "/home/davideoliveira/Programming/AIProjects/AnomalyDetection/data/processed/"


def load_data(file_name):
    """Carrega os dados""" 
    return pd.read_csv(RAW_PATH + file_name)

def load_variables_names():
    """Carrega o dicionário de nomes das variáveis."""

    with open(RAW_PATH + "variable_names.json", "r") as f:
        variable_names = json.load(f)

    return variable_names


def rename_columns(df):
    mapping = load_variables_names()
    mapping_upper = {k.upper(): v for k, v in mapping.items()}

    new_columns = {
        col: mapping_upper[col.upper()]
        for col in df.columns
        if col.upper() in mapping_upper
    }

    return df.rename(columns=new_columns)


def drop_unnecessary_columns(df, drop_fault):
    if drop_fault:
        return df.drop(columns=["simulationRun", "sample", "faultNumber"])
    else:
        return df.drop(columns=["simulationRun", "sample"])


def preprocess(file_name, drop_fault=True):
    """Carrega os dados, renomeia as colunas, dropa colunas desnecessárias e salva os dados processados."""

    print("Carregando os dados...")
    df = load_data(file_name)
    print(f"Done! size = {len(df)}")

    # renomeia as variaveis do dataset de treino
    print("Renomeando as colunas...")
    df = rename_columns(df)
    print(f"Done! size = {len(df)}")

    # remocao de colunas desnecessárias
    print("Removendo colunas desnecessárias...")
    if drop_fault:
        df = drop_unnecessary_columns(df, drop_fault)
        print(f"Done! size = {len(df)}")

        # salva como csv
        print("Salvando os dados processados...")
        df.to_csv(PROCESSED_PATH + file_name.replace(".csv", "_NoFault_Proc.csv"), index=False)
    else:
        df = drop_unnecessary_columns(df, drop_fault)
        print(f"Done! size = {len(df)}")

        # salva como csv
        print("Salvando os dados processados...")
        df.to_csv(PROCESSED_PATH + file_name.replace(".csv", "_Fault_Proc.csv"), index=False)

    print("Dados processados e salvos com sucesso!") 


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Uso: python process.py <file_name>")
        sys.exit(1)

    preprocess(sys.argv[1], drop_fault=False)