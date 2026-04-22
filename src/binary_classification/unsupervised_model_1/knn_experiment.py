import os
import gc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, precision_score
from src.binary_classification.unsupervised_model_1.evaluate import prepare_test_data, SAMPLE_FRAC


print("Carregando dataset...")
df_train = pd.read_csv("../data/processed/TEP_FaultFree_Training_Processed.csv")

# use only 5% of the data
print(f"Tamanho original: {len(df_train)}")
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=42)
print(f"Tamanho reduzido para o treino: {len(df_train)}")

print("Normalizando...")
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)


knn = NearestNeighbors(n_neighbors=6)
print("Modelo Carregado")

print("Treinando modelo...")
knn.fit(X_train)

print("Calculating anomaly score on the training dataset")
distances_train, _ = knn.kneighbors(df_train)
anomaly_score_train = distances_train.mean(axis=1)

del df_train
del X_train
gc.collect()

print("Loading test data...")
df_test = prepare_test_data()

y_true = df_test["Ground_Truth"].values
X_test = df_test.drop(columns=["Ground_Truth"])

X_test = scaler.fit_transform(X_test)

del df_test
gc.collect()

print("Calculating anomaly score on the test dataset")
distances, _ = knn.kneighbors(X_test)
anomaly_score = distances.mean(axis=1)

threshold = np.percentile(anomaly_score_train, 95)

y_pred = (anomaly_score > threshold).astype(int)

print(classification_report(y_true, y_pred))
print(precision_score(y_true, y_pred))