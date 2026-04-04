import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.anomaly import load_model, predict_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve
from unsupervised_model_1.process import RAW_PATH, PROCESSED_PATH, rename_columns, drop_unnecessary_columns
import os
import gc 
import numpy as np

# --- CONFIGURAÇÕES ---
PATH_MODELS = "/home/davideoliveira/Programming/AIProjects/AnomalyDetection/models/"
PATH_RESULTS = "/home/davideoliveira/Programming/AIProjects/AnomalyDetection/docs/results/"
FILE_TEST_CLEAN = "TEP_FaultFree_Testing_Processed.csv"
FILE_TEST_FAULTY_RAW = "TEP_Faulty_Testing.csv"

SAMPLE_FRAC = 0.05 


def reduce_mem_usage(df):
    """Itera pelas colunas e converte float64 para float32 para economizar RAM."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memória inicial: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            if str(col_type).startswith('int'):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = df[col].astype(np.float32) 

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memória final: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% redução)')
    return df


def prepare_test_data():
    print("--- Preparando Dados de Teste ---")
    
    # 1. Carregar Teste Normal
    print("Carregando Normal...")
    df_clean = pd.read_csv(os.path.join(PROCESSED_PATH, FILE_TEST_CLEAN))
    
    # Amostragem e Otimização IMEDIATAS
    df_clean = df_clean.sample(frac=SAMPLE_FRAC, random_state=42)
    df_clean['Ground_Truth'] = 0
    df_clean = reduce_mem_usage(df_clean)

    # 2. Carregar Teste Falhas (O mais pesado)
    print("Carregando Falhas (Isso pode demorar)...")
    df_faulty = pd.read_csv(os.path.join(RAW_PATH, FILE_TEST_FAULTY_RAW))
    
    # Lógica de Ground Truth (antes de processar/amostrar para garantir consistência)
    # A falha ocorre após sample 160
    df_faulty['Ground_Truth'] = (df_faulty['sample'] > 160).astype(int)
    
    # Processamento
    df_faulty = rename_columns(df_faulty)
    df_faulty = drop_unnecessary_columns(df_faulty)
    
    # Amostragem e Otimização
    df_faulty = df_faulty.sample(frac=SAMPLE_FRAC, random_state=42)
    df_faulty = reduce_mem_usage(df_faulty)

    # 3. Unir
    print("Concatenando...")
    df_test = pd.concat([df_clean, df_faulty], ignore_index=True)
    
    # Limpar variáveis antigas da memória explicitamente
    del df_clean
    del df_faulty
    gc.collect()

    print(f"Dataset de teste final: {len(df_test)} linhas.")
    print(f"Distribuição: {df_test['Ground_Truth'].value_counts().to_dict()}")
    
    return df_test


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predito (PyCaret)')
    plt.ylabel('Real (Ground Truth)')
    plt.title(f'Matriz de Confusão: {model_name}')
    plt.savefig(os.path.join(PATH_RESULTS, f"cm_{model_name}.png"))
    plt.close()


def plot_roc_curves(results_dict):
    plt.figure(figsize=(10, 8))
    for model_name, data in results_dict.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_scores'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparação ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PATH_RESULTS, "roc_comparison.png"))
    plt.close()


def plot_pr_curves(results_dict):
    """Plota a curva Precision-Recall e calcula o limiar ideal para o KNN."""
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results_dict.items():
        # Calcula os pontos da curva
        precision, recall, thresholds = precision_recall_curve(data['y_true'], data['y_scores'])
        
        # Plota a linha do modelo
        plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
        
        # --- BÔNUS: Descobrindo o Threshold Mágico para o KNN ---
        if model_name == "knn":
            # Queremos encontrar qual é o threshold para ter pelo menos 90% de Recall
            # O array de recall vai diminuindo, vamos achar o valor mais próximo de 0.90
            idx = (np.abs(recall - 0.90)).argmin() 
            
            # O array de thresholds é 1 item menor que recall/precision, então ajustamos o índice
            idx_thresh = min(idx, len(thresholds) - 1)
            best_thresh = thresholds[idx_thresh]
            
            print(f"\n[DICA DE OURO] Para o modelo KNN:")
            print(f" -> Se você quiser ~90% de Recall...")
            print(f" -> Altere seu código para usar o threshold (limiar) de: {best_thresh:.4f}")
            print(f" -> A Precisão cairá de 98% para aprox: {precision[idx]:.4f}")
            print("-" * 40)
            
            # Marca esse ponto no gráfico com um 'X' vermelho
            plt.plot(recall[idx], precision[idx], 'rX', markersize=10, 
                     label='Ponto Ideal KNN (~90% Recall)')

    plt.xlabel('Recall (Sensibilidade - Pegar todas as falhas)')
    plt.ylabel('Precision (Precisão - Evitar falsos alarmes)')
    plt.title('Comparação Precision-Recall - Detecção de Anomalias TEP')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(PATH_RESULTS, "pr_comparison.png"))
    plt.close()
    print("Gráfico Precision-Recall salvo na pasta results.")


def evaluate():
    os.makedirs(PATH_RESULTS, exist_ok=True)

    # Carregar dados (já amostrados e otimizados)
    df_test = prepare_test_data()
    
    # Separar labels para evitar passar para o PyCaret desnecessariamente
    y_true_full = df_test['Ground_Truth'].values

    data_for_prediction = df_test.drop(columns=['Ground_Truth'])

    models_list = ["iforest", "pca", "knn", "svm"]
    results_summary = []
    roc_data = {}

    for m in models_list:
        print(f"\n--- Avaliando Modelo: {m} ---")
        
        try:
            # Carregar Modelo
            model_path = os.path.join(PATH_MODELS, f"{m}_model")
            loaded_model = load_model(model_path)
            
            # Previsão
            # predict_model retorna o dataframe original + colunas novas.
            # Isso dobra a memória. Vamos pegar só o retorno necessário.
            predictions = predict_model(loaded_model, data=data_for_prediction)
            
            y_pred = predictions['Anomaly']
            y_scores = predictions['Anomaly_Score']

            # Métricas
            prec = precision_score(y_true_full, y_pred)
            rec = recall_score(y_true_full, y_pred)
            f1 = f1_score(y_true_full, y_pred)
            
            print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            
            results_summary.append({
                "Model": m, "Precision": prec, "Recall": rec, "F1-Score": f1
            })
            
            # Guardar dados para ROC (apenas scores, para economizar)
            roc_data[m] = {'y_true': y_true_full, 'y_scores': y_scores.values}

            # Plotar CM
            plot_confusion_matrix(y_true_full, y_pred, m)

            # Limpar memória do modelo e das predições
            del loaded_model
            del predictions
            gc.collect()

        except Exception as e:
            print(f"Erro ao processar {m}: {e}")

    # Salvar resultados finais
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(PATH_RESULTS, "metrics_summary.csv"), index=False)
    print("\nResumo das métricas salvo.")
    print(df_results)

    plot_roc_curves(roc_data)

    plot_pr_curves(roc_data)

if __name__ == "__main__":
    evaluate()