import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2
from src.binary_classification.unsupervised_model_1.process import PROCESSED_PATH, preprocess


df_noc = pd.read_csv(PROCESSED_PATH + "TEP_FaultFree_Training_NoFault_Proc.csv",
                     dtype=np.float32)

scaler = StandardScaler()
X_noc = scaler.fit_transform(df_noc).astype(np.float32)

# PCA
pca_full = PCA()
pca_full.fit(X_noc)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = np.argmax(cumvar >= 0.95) + 1

pca = PCA(n_components=n_comp)
T_scores = pca.fit_transform(X_noc)

P = pca.components_.T.astype(np.float32)
Lambda = pca.explained_variance_.astype(np.float32)
Lambda_inv = np.diag(1.0 / Lambda)

# ================= T² =================
T2_train = np.sum((T_scores @ Lambda_inv) * T_scores, axis=1)

T2_limit = chi2.ppf(0.99, df=n_comp)

# ================= Q (SPE) =================
X_hat = T_scores @ P.T
residual = X_noc - X_hat
Q_train = np.sum(residual**2, axis=1)

Q_limit = np.percentile(Q_train, 99)


def detect_fault(X_new):
    X_scaled = scaler.transform(X_new).astype(np.float32)

    T_new = X_scaled @ P
    T2_new = np.sum((T_new @ Lambda_inv) * T_new, axis=1)

    X_hat = T_new @ P.T
    residual = X_scaled - X_hat
    Q_new = np.sum(residual**2, axis=1)

    fault_flag = (T2_new > T2_limit) | (Q_new > Q_limit)

    return fault_flag, T2_new, Q_new, residual


def m_rbc_contribution(residual):
    contribution = residual**2

    var = np.var(X_noc, axis=0)
    contribution = contribution / (var + 1e-8)

    return contribution


def detection_metrics(y_true, fault_flag):
    y_binary = (y_true != 0).astype(int)
    y_pred = fault_flag.astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()

    FDR = tp / (tp + fn)
    FAR = fp / (fp + tn)
    ACC = (tp + tn) / (tp + tn + fp + fn)

    print(f"FDR: {FDR:.4f}")
    print(f"FAR: {FAR:.4f}")
    print(f"Accuracy: {ACC:.4f}")

    return FDR, FAR, ACC


def plot_statistics(T2, Q):
    plt.figure()
    plt.plot(T2)
    plt.axhline(T2_limit, linestyle='--')
    plt.title("Hotelling T²")
    plt.show()

    plt.figure()
    plt.plot(Q)
    plt.axhline(Q_limit, linestyle='--')
    plt.title("Q Statistic (SPE)")
    plt.show()


df_fault_train = pd.read_csv(
    PROCESSED_PATH + "TEP_Faulty_Training_Fault_Proc.csv",
    dtype=np.float32
)

X_fault_train = df_fault_train.drop(columns=["faultNumber"])
y_fault_train = df_fault_train["faultNumber"]

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_fault_train, y_fault_train)

# test

df_test = pd.concat([
    pd.read_csv(PROCESSED_PATH + "TEP_FaultFree_Testing_Fault_Proc.csv", dtype=np.float32),
    pd.read_csv(PROCESSED_PATH + "TEP_Faulty_Testing_Fault_Proc.csv", dtype=np.float32)
], ignore_index=True)

X_test = df_test.drop(columns=["faultNumber"])
y_true = df_test["faultNumber"].values

fault_flag, T2_test, Q_test, residual = detect_fault(X_test)

detection_metrics(y_true, fault_flag)

plot_statistics(T2_test, Q_test)

contrib = m_rbc_contribution(residual[fault_flag])

ranking = np.argsort(contrib.mean(axis=0))[::-1]
top_k = 10
selected_variables = X_test.columns[ranking[:top_k]]

X_selected = X_test[fault_flag][selected_variables]
pred_fault_class = rf.predict(X_selected)

print(classification_report(y_true[fault_flag], pred_fault_class))