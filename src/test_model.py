import sys
sys.path.append('/home/msiau/workspace/asdf/models')

import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

from custom_dataset import *
from NN_architectures.old_school import GrainClassifierOldSchool
from NN_architectures.sh import GrainClassifierSH
from utils.classification import *
from NAG import get_test_meshes_pkl

pretreatment = "sh"
experiment = "Bere"
idd = "10"
model_path = f"/home/msiau/workspace/asdf/models/saved/{pretreatment}/{experiment}{idd}"

# Load dataset
#dataset_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_old_school/{experiment}"
dataset_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_SH_L50_xyz/{experiment}"
test_array, labels, classes = get_test_meshes_pkl(dataset_path)

test_dataset = CustomDatasetFromArray(test_array, labels)
n_classes = len(np.unique(test_dataset.labels))
l_max = 50

# DataLoaders
batch_size = 1
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

GrainClassifierOldSchool
#model = GrainClassifierOldSchool(n_classes = n_classes)
model = GrainClassifierSH(l_max = l_max, n_classes = n_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

def confusion_matrix(dataloader, model_nn, n):
    cm = np.zeros((n, n), dtype=np.int_)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model_nn(inputs)
        _, predictions = torch.max(outputs.data, 1)
        label = int(labels[0])
        prediction = int(predictions[0])
        cm[label, prediction] += 1
    print("\nConfusion matrix:")
    print(cm)
    ncm = cm.astype(np.float64)
    acc = 0
    for i in range(n):
        ncm[i] /= np.sum(ncm[i])
        acc += cm[i,i]
    acc /= np.sum(np.sum(cm))
    print("\nNormalized confusion matrix:")
    print(ncm)
    print("\nAccuracy:")
    print(acc)

confusion_matrix(test_loader, model, n_classes)
for i, cl in enumerate(classes):
    print(f"{i}. {cl}")