import sys
sys.path.append('/home/msiau/workspace/asdf/models')

import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import config

from custom_dataset import CustomDatasetFromPickle
from NN_architectures.old_school import GrainClassifierOldSchool
from utils.classification import *

def train_one_epoch(dataloader):
    running_loss = 0.
    for i, data in enumerate(dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        #print("inputs, labels")
        #print(inputs[0], labels[0])
        #print("outputs")
        #print(outputs[0])
        #print("labels")
        #print(labels[0])
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()
    #print("Total loss: ", running_loss)
    return running_loss

def test_model(dataloader):
    total_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        _, predictions = torch.max(outputs.data, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)  # Número total de ejemplos

    # Calcula la precisión como el número de predicciones correctas dividido por el total
    accuracy = correct / total
        
    return total_loss, accuracy

dataset_path = "/home/msiau/data/tmp/jesmoris/old_school"
n_epochs = 3000

# Load dataset
dataset = CustomDatasetFromPickle(dataset_path)
adjust_labels(dataset, list_dundee_orkney)
n_classes = len(np.unique(dataset.labels.cpu()))

# Train/test split
batch_size = 16
train_size = 0.7
folds = stratified_random_split(dataset=dataset, train_size=train_size, n_splits=5)

with open("/home/msiau/workspace/asdf/src/logs/train/old_school/cross_validation/landraces_fold", 'w') as file:
        file.write(" ")

for i_fold, fold in enumerate(folds):
    save_path = f"/home/msiau/workspace/asdf/models/saved/old_school/cross_validation/landraces_fold_{i_fold}"
    log_path = f"/home/msiau/workspace/asdf/src/logs/train/old_school/cross_validation/landraces_fold_{i_fold}"
    train_index = fold[0]
    test_index = fold[1]
    data = dataset.data
    labels = dataset.labels
    train_dataset = CustomDatasetFromArray(data[train_index], labels[train_index])
    test_dataset = CustomDatasetFromArray(data[test_index], labels[test_index])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GrainClassifierOldSchool(n_classes)

    # Optimizers specified in the torch.optim package
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    with open(log_path, 'w') as file:
        file.write("Epoch,train_loss,test_loss,accuracy\n")

    max_accuracy = 0
    i_max_accuracy = 0
    for i in range(n_epochs):
        train_loss = train_one_epoch(train_loader)
        test_loss, accuracy = test_model(test_loader)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            i_max_accuracy = i
            print("MAX ACCURACY")
            torch.save(model.state_dict(), save_path)
        print("Epoch: " + "%-3i" % i + ", train loss: " + str(train_loss) + ", test loss: " + str(test_loss), ", accuracy: " + str(accuracy))
        log_string(log_path, f"{i},{train_loss},{test_loss},{accuracy}\n")
    log_string(log_path, f"\n\nMaximum Accuracy: {max_accuracy} in epoch {i_max_accuracy}\n")
    log_string(log_path, "\nParameters:\n")
    log_string(log_path, f"- Batch size = {batch_size}\n")
    log_string(log_path, f"- Train size = {len(train_dataset)}, {round(train_size*100)}%\n")
    log_string(log_path, f"- Test size = {len(test_dataset)}, {round(100* (1-train_size))}%\n")
    log_string(log_path, f"- Learning rate = {learning_rate}\n")
    log_string(log_path, f"- Momentum = {momentum}\n")
    log_string(log_path, f"\nFOLD {i_fold}:\n")
    log_string(log_path, "\nTrain indices:\n")
    log_string(log_path, ''.join(str(x) + "," for x in train_index))
    log_string(log_path, "\nTest indices:\n")
    log_string(log_path, ''.join(str(x) + "," for x in test_index))
    
    log_string(log_path[:-2], f"\nFOLD {i_fold}:\n")
    log_string(log_path[:-2], "\nTrain indices:\n")
    log_string(log_path[:-2], ''.join(str(x) + "," for x in train_index))
    log_string(log_path[:-2], "\nTest indices:\n")
    log_string(log_path[:-2], ''.join(str(x) + "," for x in test_index))
    log_string(log_path[:-2], f"\n\nMaximum Accuracy: {max_accuracy} in epoch {i_max_accuracy}\n")


    print(max_accuracy)

