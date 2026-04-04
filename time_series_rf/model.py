import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report
import gc
import os

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 20
RAND_SEED = 42
# Das 500 sims de falha, vamos usar apenas 24 no total (para manter 50/50)
TOTAL_FAULTY_SIMS_TO_USE = 24 
N_JOBS = -1 # Usar todos os núcleos

INPUT_DIR = Path("../data/raw/")
FAULT_FREE_TRAIN_PATH = INPUT_DIR / "TEP_FaultFree_Training.csv"
FAULTY_TRAIN_PATH = INPUT_DIR / "TEP_Faulty_Training.csv"

MODELS_DIR = Path("models/")
if not MODELS_DIR.exists():
    MODELS_DIR.mkdir(parents=True)

MODEL_NAME = "Rnd_Forest_TS_Bin.pkl"


def get_memory_safe_windows(file_path, window_size=WINDOW_SIZE, is_faulty=False, sim_ids_to_keep=None):
    """
    Lê o arquivo, filtra pelas simulações desejadas e gera janelas em float32.
    """
    print(f"Lendo arquivo: {file_path.name}...")
    df = pd.read_csv(file_path)
    
    # Coluna de amostra no treino é 'sample' conforme seu código anterior
    cols_to_drop = ['simulationRun', 'sample', 'faultNumber']
    
    # Filtro Crítico por Simulação
    if sim_ids_to_keep is not None:
        df = df[df['simulationRun'].isin(sim_ids_to_keep)]
    
    if df.empty:
        raise ValueError(f"Nenhuma simulação encontrada em {file_path.name} com os IDs fornecidos.")

    # Identifica colunas de sensores
    sensor_cols = [c for c in df.columns if c not in cols_to_drop]
    n_features = len(sensor_cols)
    
    # Calculando tamanho final para pre-alocação
    grouped = df.groupby('simulationRun')
    total_windows = sum(len(group) - window_size + 1 for _, group in grouped)
    
    print(f"  > Gerando {total_windows} janelas (Falha: {is_faulty})...")
    
    # Pre-alocação em float32
    X_final = np.empty((total_windows, window_size * n_features), dtype=np.float32)
    y_final = np.empty(total_windows, dtype=np.int8)
    
    curr_idx = 0
    for sim_id, group in grouped:
        sensors = group[sensor_cols].values.astype(np.float32)
        
        # Strides (sem cópia de memória)
        shape = (sensors.shape[0] - window_size + 1, window_size, sensors.shape[1])
        strides = (sensors.strides[0], sensors.strides[0], sensors.strides[1])
        windows = np.lib.stride_tricks.as_strided(sensors, shape=shape, strides=strides)
        
        # Flatten e preenchimento
        n_win = windows.shape[0]
        X_final[curr_idx : curr_idx + n_win] = windows.reshape(n_win, -1)
        y_final[curr_idx : curr_idx + n_win] = 1 if is_faulty else 0
        
        curr_idx += n_win

    del df
    gc.collect()
    return X_final, y_final

def prepare_split_ids(file_path, max_sims=None, train_ratio=0.8):
    """
    Lê apenas a coluna simulationRun, seleciona a quantidade desejada
    e faz o split 80/20 dos IDs.
    """
    # Lê apenas a coluna necessária para economizar RAM
    df_ids = pd.read_csv(file_path, usecols=['simulationRun'])
    unique_ids = df_ids['simulationRun'].unique()
    
    if max_sims:
        # Se for o dataset de falha, limitamos a 24
        unique_ids = unique_ids[unique_ids <= max_sims]
        
    np.random.seed(RAND_SEED)
    np.random.shuffle(unique_ids)
    
    n_train = int(len(unique_ids) * train_ratio)
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:]
    
    del df_ids
    gc.collect()
    return train_ids, val_ids

# --- FLUXO PRINCIPAL ---

np.random.seed(RAND_SEED)

print("="*30)
print("INICIANDO PREPARAÇÃO (TRAIN/VAL SPLIT)")
print("="*30)

# 1. Preparar IDs para o Dataset NORMAL (FaultFree) - Usa as 500 sims
# IDs: ~400 treino, ~100 validação
train_ids_norm, val_ids_norm = prepare_split_ids(FAULT_FREE_TRAIN_PATH, train_ratio=0.8)

# 2. Preparar IDs para o Dataset com FALHA (Faulty) - Limita a 24 sims
# IDs: ~19 treino, ~5 validação
train_ids_fault, val_ids_fault = prepare_split_ids(FAULTY_TRAIN_PATH, max_sims=TOTAL_FAULTY_SIMS_TO_USE, train_ratio=0.8)

print(f"IDs Normais: {len(train_ids_norm)} treino, {len(val_ids_norm)} val")
print(f"IDs Falha:   {len(train_ids_fault)} treino, {len(val_ids_fault)} val")

# 3. Gerar Janelas de TREINO
print("\n--- Gerando Dados de TREINO ---")
X_train_norm, y_train_norm = get_memory_safe_windows(FAULT_FREE_TRAIN_PATH, is_faulty=False, sim_ids_to_keep=train_ids_norm)
X_train_fault, y_train_fault = get_memory_safe_windows(FAULTY_TRAIN_PATH, is_faulty=True, sim_ids_to_keep=train_ids_fault)

X_train = np.concatenate([X_train_norm, X_train_fault], axis=0)
y_train = np.concatenate([y_train_norm, y_train_fault], axis=0)

del X_train_norm, y_train_norm, X_train_fault, y_train_fault
gc.collect()

# 4. Gerar Janelas de VALIDAÇÃO
print("\n--- Gerando Dados de VALIDAÇÃO ---")
X_val_norm, y_val_norm = get_memory_safe_windows(FAULT_FREE_TRAIN_PATH, is_faulty=False, sim_ids_to_keep=val_ids_norm)
X_val_fault, y_val_fault = get_memory_safe_windows(FAULTY_TRAIN_PATH, is_faulty=True, sim_ids_to_keep=val_ids_fault)

X_val = np.concatenate([X_val_norm, X_val_fault], axis=0)
y_val = np.concatenate([y_val_norm, y_val_fault], axis=0)

del X_val_norm, y_val_norm, X_val_fault, y_val_fault
gc.collect()

# 5. Embaralhar Treino (Crucial)
print("\nEmbaralhando dataset de treino...")
X_train, y_train = shuffle(X_train, y_train, random_state=RAND_SEED)

# 6. Treinamento
print(f"\nTreinando Random Forest (Total Train Windows: {X_train.shape[0]})...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.5, # RAM Save
    max_features='sqrt',
    n_jobs=N_JOBS,
    random_state=RAND_SEED,
    verbose=1
)
rf.fit(X_train, y_train)

# 7. Avaliação Rápida na VALIDAÇÃO
print("\n--- Avaliação no Conjunto de VALIDAÇÃO (Hold-out) ---")
y_val_pred = rf.predict(X_val)
print(classification_report(y_val, y_val_pred, target_names=['Normal', 'Falha']))
v_f1 = f1_score(y_val, y_val_pred)
print(f"F1-Score na Validação: {v_f1:.4f}")

# 8. Salvar Modelo
save_path = MODELS_DIR / MODEL_NAME
print(f"\nSalvando modelo em: {save_path}")
joblib.dump(rf, save_path)

print("\nFim do treinamento e validação.")