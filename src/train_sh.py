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
from NN_architectures.sh import GrainClassifierSH
from utils.classification import *

experiment = "Bere"
idd = "10"
save_path = f"/home/msiau/workspace/asdf/models/saved/sh/{experiment}{idd}"
log_path = f"/home/msiau/workspace/asdf/src/logs/train/sh/{experiment}{idd}.csv"
img_path = f"/home/msiau/workspace/asdf/src/logs/train/sh/{experiment}{idd}.png"

# Load dataset
dataset_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_SH_L50_xyz/{experiment}"
train_dataset, validation_dataset, test_dataset = train_val_test_split(dataset_path)
n_classes = len(np.unique(train_dataset.labels))
l_max = 50

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

model = GrainClassifierSH(l_max = l_max, n_classes = n_classes)

# Optimizers specified in the torch.optim package
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.0005
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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

with open(log_path, 'w') as file:
    file.write("Epoch,train_loss,validation_loss,accuracy\n")

max_accuracy = 0
i_max_accuracy = 0
accuracies = []
for i in range(10000):
    # Train one epoch
    train_loss = train_one_epoch(train_loader)
    # Test model with validation set
    validation_loss, accuracy = test_model(validation_loader)
    accuracies.append(accuracy)
    if max_accuracy < accuracy:
        max_accuracy = accuracy
        i_max_accuracy = i
        print("MAX ACCURACY")
        print("Epoch: " + "%-3i" % i + ", train loss: " + str(train_loss) + ", validation loss: " + str(validation_loss), ", accuracy: " + str(accuracy))
        torch.save(model.state_dict(), save_path)
        log_string(log_path, f"{i},{train_loss},{validation_loss},{accuracy}\n")
        continue

    print("Epoch: " + "%-3i" % i + ", train loss: " + str(train_loss) + ", validation loss: " + str(validation_loss), ", accuracy: " + str(accuracy))
    log_string(log_path, f"{i},{train_loss},{validation_loss},{accuracy}\n")
log_string(log_path, f"\n\nMaximum Accuracy: {max_accuracy} in epoch {i_max_accuracy}\n")
log_string(log_path, "\nParameters:\n")
log_string(log_path, f"- Batch size = {batch_size}\n")
log_string(log_path, f"- Train size = {len(train_dataset)}\n")
log_string(log_path, f"- Validation size = {len(validation_dataset)}\n")
log_string(log_path, f"- Learning rate = {learning_rate}\n")
log_string(log_path, f"- Momentum = {momentum}\n")

print(max_accuracy)

plot_accuracy_epoch(accuracies, save_path=img_path, kernel_len=15)
