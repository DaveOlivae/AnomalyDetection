import pandas as pd
from pycaret.anomaly import *


def run_experiment():

    df_train = pd.read_csv("../data/processed/TEP_FaultFree_Training_Processed.csv")

    # use only 5% of the data
    print(f"Tamanho original: {len(df_train)}")
    df_train = df_train.sample(frac=0.05, random_state=42)
    print(f"Tamanho reduzido para o treino: {len(df_train)}")

    model = setup(
        data=df_train,
        normalize=True,
        session_id=42,
        verbose=False
    )

    models = ["iforest", "pca", "knn", "svm"]

    trained_models = {}

    for m in models:
        print(f"Treinando o modelo {m}...")
        model = create_model(m)
        trained_models[m] = model
        save_model(model, f"/home/davideoliveira/Programming/AIProjects/AnomalyDetection/models/{m}_model")
        print(f"Modelo {m} treinado e salvo com sucesso!")
    
    print("Todos os modelos foram treinados e salvos com sucesso!")

    return trained_models


if __name__ == "__main__":
    run_experiment() 
