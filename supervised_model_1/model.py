from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from data_loading import load_data, DATA_PATH, RAND_SEED

MODELS_PATH = "models/"
RESULTS_PATH = Path("results/")

file_path = Path(DATA_PATH)

if file_path.exists():
    print("Carregando dataset...")
    df_train = pd.read_csv(file_path)

    X = df_train.drop("faultNumber", axis=1).astype("float32")
    y = df_train["faultNumber"].astype("int8")
else:
    print("Criando dataset...")

    X, y = load_data()

print("Splitting train and eval dataset...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RAND_SEED,
    stratify=y
)
print(f"X_train = {len(X_train)}, X_val = {len(X_val)}, y_train = {len(y_train)}, y_val = {len(y_val)}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1
)

print("Training XGBOOST model...")
xgb_model.fit(X_train, y_train)

model_file_path = RESULTS_PATH / "xgb.json"
xgb_model.save_model(str(model_file_path))

# evaluation

acc_train = xgb_model.score(X_train, y_train)

# predictions
pred_val = xgb_model.predict(X_val)

# confustion matrix
cm = confusion_matrix(y_val, pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_img_path = str(RESULTS_PATH / "Confusion_Matrix_01.png")

disp.plot().figure_.savefig(cm_img_path, dpi=300)

print(f"Matriz de Confusão salva em {cm_img_path}!")

# classification report
report = classification_report(y_val, pred_val)
cl_repo_path = RESULTS_PATH / "Classification_Report_01.txt"

with open(cl_repo_path, "w", encoding="utf-8") as f:
    f.write("=== Relatório de Classificação ===\n\n")
    f.write(f" Accuracy no train set = {acc_train}\n\n")
    f.write(report)

print(f"Relatório salvo com sucesso em {cl_repo_path}")
