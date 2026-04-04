
# Possible Datasets

CWRU Bearing Dataset: https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets

Tennessee Eastman Process Simulation Dataset: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset

Time-Series of Industrial Boiler Operations: https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations

## Tennessee Eastman Process

### Variáveis:
- 41 variáveis medidas (?)
- 11 variáveis manipuladas (medidas que o operador pode manipular para garantir que o processo esteja sob controle)

total de 52 variáveis

### Falhas
20 tipos de falhas diferentes

### Dataset
O Dataset contém dados "fault-free" (Operação normal) e "faulty" (Contém 20 tipos de falhas diferentes)

Logo, temos 4 arquivos .RData: FaultFree_Testing, FaultFree_Training, Faulty_Testing e Faulty_Training

#### Treino

amostras coletadas a cada 3 min por 25 horas (500 amostras)

**Fault-Free Training Length:** 250,000 lines
**Faulty Training Length:** 5,000,000 lines (250,000 por falta, em cada simulacao ha 500 samples de cada falha)

#### Teste

amostras coletadas a cada 3 min por 48 horas (500 simulacoes, 960 amostras cada)

**Fault-Free Testing Length:** 480,000 lines
**Faulty Testing Length:** 9,600,000 lines  

# Unsupervised Model 1 - Pipeline

Quatro algoritmos serão utilizados: isolation forest, pca, k-nearest neighbours e svm.

Como knn e svm tem dificuldade em processar grandes volumes de dados, apenas uma porcentagem do dataset de treino normal será utilizado.