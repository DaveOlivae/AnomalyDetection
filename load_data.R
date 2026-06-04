load("../data/raw/dataverse_files/TEP_FaultFree_Testing.RData")
write.csv(fault_free_testing, "../data/raw/TEP_FaultFree_Testing.csv", row.names = FALSE)

load("../data/raw/dataverse_files/TEP_FaultFree_Training.RData")
write.csv(fault_free_training, "../data/raw/TEP_FaultFree_Training.csv", row.names = FALSE)

load("../data/raw/dataverse_files/TEP_Faulty_Testing.RData")
write.csv(faulty_testing, "../data/raw/TEP_Faulty_Testing.csv", row.names = FALSE)

load("../data/raw/dataverse_files/TEP_Faulty_Training.RData")
write.csv(faulty_training, "../data/raw/TEP_Faulty_Training.csv", row.names = FALSE)