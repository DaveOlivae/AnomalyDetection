import numpy as np
import pandas as pd

# config
RAND_SEED = 42
N_SAMPLES = 100
N_SIMS = 500
ROWS_PER_SIM_FAULTFREE = 500
ROWS_PER_SIM_FAULTY = 10_000
FAULT_FREE_PATH = "../data/raw/TEP_FaultFree_Training.csv"
FAULTY_PATH = "../data/raw/TEP_Faulty_Training.csv"
DATA_PATH = "data/tep_train_100sims.csv"


def load_data():
    # select random simulations
    np.random.seed(RAND_SEED)
    selected_simulations = np.random.choice(range(N_SIMS), size=N_SAMPLES, replace=False)
    selected_simulations = sorted(selected_simulations)
    print(f"Chosen simulations: {selected_simulations}")

    # faultfree
    dfs_faultfree = []

    print("Reading Fault-Free Dataset...")
    for sim in selected_simulations:
        start = sim * ROWS_PER_SIM_FAULTFREE

        df_chunk = pd.read_csv(
            FAULT_FREE_PATH,
            skiprows=range(1, start + 1),
            nrows=ROWS_PER_SIM_FAULTFREE
        )

        dfs_faultfree.append(df_chunk)

    df_faultfree = pd.concat(dfs_faultfree, ignore_index=True)
    del dfs_faultfree

    print(f"Length of Fault-Free Dataframe = {len(df_faultfree)}")

    # faulty
    dfs_faulty = []

    print("Reading Faulty Dataset...")
    for sim in selected_simulations:
        start = sim * ROWS_PER_SIM_FAULTY

        df_chunk = pd.read_csv(
            FAULTY_PATH,
            skiprows=range(1, start + 1),
            nrows=ROWS_PER_SIM_FAULTY
        )

        dfs_faulty.append(df_chunk)

    df_faulty = pd.concat(dfs_faulty, ignore_index=True)
    del dfs_faulty

    print(f"Length of Faulty dataset = {len(df_faulty)}")

    print("Concatenating datasets...")
    df_train = pd.concat([df_faultfree, df_faulty], ignore_index=True)
    print(f"Length of final dataset = {len(df_train)}")
    del df_faultfree
    del df_faulty

    df_train = df_train.drop(columns=["simulationRun", "sample"])

    df_train.to_csv(DATA_PATH, index=False)

    X_train = df_train.drop("faultNumber", axis=1).astype("float32")
    y_train = df_train["faultNumber"].astype("int8")

    return X_train, y_train