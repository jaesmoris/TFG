import sys
sys.path.append('/home/msiau/workspace/asdf/models')

import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

from custom_dataset import CustomDatasetFromPickle
from NN_architectures.old_school import GrainClassifierOldSchool
from NN_architectures.sh import GrainClassifierSH
from utils.classification import *

#dataset_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L50"
dataset_path = "/home/msiau/data/tmp/jesmoris/old_school"
#model_path = "/home/msiau/workspace/asdf/models/saved/old_school/2row_6row"
#model_path = "/home/msiau/workspace/asdf/models/saved/old_school/Dundee_Orkney"
#model_path = "/home/msiau/workspace/asdf/models/saved/old_school/Landrace"
#model_path = "/home/msiau/workspace/asdf/models/saved/sh/2row_6row_L50"
#model_path = "/home/msiau/workspace/asdf/models/saved/sh/dundee_orkney_L50"
model_path = "/home/msiau/workspace/asdf/models/saved/sh/landraces_L50"


# Load dataset
dataset = CustomDatasetFromPickle(dataset_path)

#adjust_labels(dataset, list_2row_6row)
#adjust_labels(dataset, list_dundee_orkney)
adjust_labels(dataset, list_landraces)

train_dataset, test_dataset = stratified_random_split_train_test(dataset, train_size=0.7)

n_classes = len(np.unique(dataset.labels))
print(f"nclasses = {n_classes}")

# Data Loader
batch_size = 1
dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
#model = GrainClassifierSH(l_max = 50, n_classes=n_classes)
#model = GrainClassifierOldSchool(n_classes)
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

confusion_matrix(dataset_loader, model, n_classes)
