import torch
import gpustat
import numpy as np
from sklearn import svm
import torch.nn.functional as F


# MoC_similarity
def moc_similarity(source_hidden_features, target_hidden_features):
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())
    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)

    moc = torch.mean(torch.max(temp_matrix, dim=0)[0]) # for partial domain adaptation
    return -moc
    

def pick_gpu_lowest_memory():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU
    