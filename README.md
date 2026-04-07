Anomaly Detection Studies

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

# Reference

Artigos que usam o TEP como estudo de caso/validação para seus modelos:

1. "A Multiagent-Based Methodology for Known and Novel Faults Diagnosis in Industrial Processes"
    
2. "Adversarial Attacks and Defenses in Fault Detection and Diagnosis: A Comprehensive Benchmark on the Tennessee Eastman Process" 
    
3. "An adaptive fault detection and root-cause analysis scheme for complex industrial processes using moving window KPCA and information geometric causal inference"
    
4. "BibMon: An open source Python package for process monitoring, soft sensing, and fault diagnosis"
    
5. "Deep convolutional neural network model based chemical process fault diagnosis"
    
6. "Explainable AI methodology for understanding fault detection results during Multi-Mode operations"
    
7. "Fault Detection of Complex Processes Using nonlinear Mean Function Based Gaussian Process Regression: Application to the Tennessee Eastman Process"
    
8. "Fault detection and diagnosis for non-linear processes empowered by dynamic neural networks"
    
9. "SensorDBSCAN: Semi-Supervised Active Learning Powered Method for Anomaly Detection and Diagnosis"
    
10. "Two-view LSTM variational auto-encoder for fault detection and diagnosis in multivariable manufacturing processes"
    
11. "XFDDC: eXplainable Fault Detection Diagnosis and Correction framework for chemical process systems"