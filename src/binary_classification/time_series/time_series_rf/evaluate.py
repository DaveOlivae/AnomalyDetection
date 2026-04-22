import pandas as pd
import numpy as np
import joblib
import os
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

# 1. Configuração de Pastas e Arquivos
OUTPUT_DIR = 'results'
MODEL_PATH = 'modelo_tep_rf.pkl' # Nome do seu arquivo salvo
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_windows_generator(file_path, window_size=20, is_faulty=False, n_sims=None):
    """
    Função para processar o teste sem estourar a RAM.
    """
    df = pd.read_csv(file_path)
    if n_sims:
        df = df[df['simulationRun'] <= n_sims]
    
    sensor_cols = [c for c in df.columns if c not in ['simulationRun', 'sample', 'faultNumber']]
    
    X_list, y_list = [], []
    
    for _, group in df.groupby('simulationRun'):
        sensors = group[sensor_cols].values.astype(np.float32)
        
        # Strides (Janelas sem cópia)
        shape = (sensors.shape[0] - window_size + 1, window_size, sensors.shape[1])
        strides = (sensors.strides[0], sensors.strides[0], sensors.strides[1])
        windows = np.lib.stride_tricks.as_strided(sensors, shape=shape, strides=strides)
        
        X_list.append(windows.reshape(windows.shape[0], -1))
        y_list.append(np.full(windows.shape[0], 1 if is_faulty else 0))
    
    del df
    gc.collect()
    return np.vstack(X_list), np.concatenate(y_list)

# --- INÍCIO DA AVALIAÇÃO ---

print(f"Lendo modelo de {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# Carregando dados de teste (limitando simulações se necessário para poupar RAM)
print("Processando dados de teste...")
X_test_norm, y_test_norm = get_windows_generator('tep_normal_test.csv', is_faulty=False, n_sims=50)
X_test_fault, y_test_fault = get_windows_generator('tep_faulty_test.csv', is_faulty=True, n_sims=10)

X_test = np.vstack([X_test_norm, X_test_fault])
y_test = np.concatenate([y_test_norm, y_test_fault])

del X_test_norm, y_test_norm, X_test_fault, y_test_fault
gc.collect()

print("Realizando predições...")
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# --- SALVANDO RESULTADOS ---

# 1. Classification Report (TXT)
print("Salvando Classification Report...")
report = classification_report(y_test, y_pred, target_names=['Normal', 'Falha'])
with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
    f.write("RELATÓRIO DE AVALIAÇÃO - TENNESSEE EASTMAN PROCESS\n")
    f.write("="*50 + "\n")
    f.write(report)
    f.write(f"\nROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    f.write(f"\nAverage Precision (PR): {average_precision_score(y_test, y_probs):.4f}")

# 2. Matriz de Confusão (PNG)
print("Gerando Matriz de Confusão...")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', ax=ax)
ax.set_title('Matriz de Confusão - TEP')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

# 3. Curva ROC (PNG)
print("Gerando Curva ROC...")
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_probs, ax=ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title(f'Curva ROC (AUC: {roc_auc_score(y_test, y_probs):.4f})')
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_auc.png'), dpi=300)
plt.close()

# 4. Curva Precision-Recall (PNG)
print("Gerando Curva PR...")
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_probs, ax=ax)
ax.set_title(f'Curva Precision-Recall (AP: {average_precision_score(y_test, y_probs):.4f})')
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'), dpi=300)
plt.close()

print(f"\nSucesso! Todos os arquivos foram salvos na pasta '{OUTPUT_DIR}'.")