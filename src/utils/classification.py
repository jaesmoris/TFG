from sklearn.model_selection import StratifiedShuffleSplit
from custom_dataset import CustomDatasetFromArray, CustomDatasetFromFold
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

########### LABEL ADJUSTMENTS ###########
# 2ROW / 6ROW adjustment
list_2row_6row = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
"""
    dataset.labels[dataset.labels == 0] = 0  # Dundee 2ROW British
    dataset.labels[dataset.labels == 1] = 0  # Dundee 2ROW Scottish
    dataset.labels[dataset.labels == 2] = 1  # Dundee 6ROW BERE Orkney
    dataset.labels[dataset.labels == 3] = 1  # Dundee 6ROW BERE Unknown
    dataset.labels[dataset.labels == 4] = 1  # Dundee 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 5] = 1  # Dundee 6ROW Faro
    dataset.labels[dataset.labels == 6] = 1  # Dundee 6ROW Scandinavian
    dataset.labels[dataset.labels == 7] = 0  # Orkney 2ROW British
    dataset.labels[dataset.labels == 8] = 0  # Orkney 2ROW Scottish
    dataset.labels[dataset.labels == 9] = 1  # Orkney 6ROW BERE Orkney
    dataset.labels[dataset.labels == 10] = 1  # Orkney 6ROW BERE Unknown
    dataset.labels[dataset.labels == 11] = 1  # Orkney 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 12] = 1  # Orkney 6ROW Scandinavian
"""
list_landraces = [0, 1, 2, 2, 2, 3, 4, 0, 1, 2, 2, 2, 4]
"""
    dataset.labels[dataset.labels == 0] = 0  # Dundee 2ROW British
    dataset.labels[dataset.labels == 1] = 1  # Dundee 2ROW Scottish
    dataset.labels[dataset.labels == 2] = 2  # Dundee 6ROW BERE Orkney
    dataset.labels[dataset.labels == 3] = 2  # Dundee 6ROW BERE Unknown
    dataset.labels[dataset.labels == 4] = 2  # Dundee 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 5] = 3  # Dundee 6ROW Faro
    dataset.labels[dataset.labels == 6] = 4  # Dundee 6ROW Scandinavian
    dataset.labels[dataset.labels == 7] = 0  # Orkney 2ROW British
    dataset.labels[dataset.labels == 8] = 1  # Orkney 2ROW Scottish
    dataset.labels[dataset.labels == 9] = 2  # Orkney 6ROW BERE Orkney
    dataset.labels[dataset.labels == 10] = 2  # Orkney 6ROW BERE Unknown
    dataset.labels[dataset.labels == 11] = 2  # Orkney 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 12] = 4  # Orkney 6ROW Scandinavian
"""
list_dundee_orkney = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
"""
    dataset.labels[dataset.labels == 0] = 0  # Dundee 2ROW British
    dataset.labels[dataset.labels == 1] = 0  # Dundee 2ROW Scottish
    dataset.labels[dataset.labels == 2] = 0  # Dundee 6ROW BERE Orkney
    dataset.labels[dataset.labels == 3] = 0  # Dundee 6ROW BERE Unknown
    dataset.labels[dataset.labels == 4] = 0  # Dundee 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 5] = 0  # Dundee 6ROW Faro
    dataset.labels[dataset.labels == 6] = 0  # Dundee 6ROW Scandinavian
    dataset.labels[dataset.labels == 7] = 1  # Orkney 2ROW British
    dataset.labels[dataset.labels == 8] = 1  # Orkney 2ROW Scottish
    dataset.labels[dataset.labels == 9] = 1  # Orkney 6ROW BERE Orkney
    dataset.labels[dataset.labels == 10] = 1  # Orkney 6ROW BERE Unknown
    dataset.labels[dataset.labels == 11] = 1  # Orkney 6ROW BERE Western Isles
    dataset.labels[dataset.labels == 12] = 1  # Orkney 6ROW Scandinavian
"""

def adjust_labels(dataset, label_list):
    # Assumed labels are sorted alphanumerically
    for index, label in enumerate(label_list):
        dataset.labels[dataset.labels == index] = label

def stratified_random_split(dataset, train_size, n_splits):
    "devuelve una lista de listas que contiene dos listas (train, test)"
    "lista[n_splits, 2, variable]"
    indices = [i for i in range(len(dataset))]
    labels = dataset.labels.cpu()
    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=42)
    sss.get_n_splits(indices, labels)
    splits = sss.split(indices, labels)
    folds = []
    for i, (train_index, test_index) in enumerate(splits):
        #print(f"Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
        folds.append([train_index, test_index])
    #print(folds)
    return folds

def stratified_random_split_train_test(dataset, train_size):
    folds = stratified_random_split(dataset, train_size, n_splits=1)
    fold = folds[0]
    train_index = fold[0]
    test_index = fold[1]
    data = dataset.data
    labels = dataset.labels
    train_dataset = CustomDatasetFromArray(data[train_index], labels[train_index])
    test_dataset = CustomDatasetFromArray(data[test_index], labels[test_index])
    return train_dataset, test_dataset

def train_test_split_from_indices(dataset, train_indices, test_indices):
    train_index = train_indices.cpu()
    test_index = test_indices.cpu()
    data = dataset.data.cpu()
    labels = dataset.labels.cpu()
    train_dataset = CustomDatasetFromArray(data[train_index], labels[train_index])
    test_dataset = CustomDatasetFromArray(data[test_index], labels[test_index])
    return train_dataset, test_dataset

def train_val_test_split(dataset_path):
    #dataset/class/fold
    train_dataset = CustomDatasetFromFold(dataset_path, "train")
    validation_dataset = CustomDatasetFromFold(dataset_path, "val")
    test_dataset = CustomDatasetFromFold(dataset_path, "test")
    return train_dataset, validation_dataset, test_dataset

def log_string(log_path : str, s : str):
    with open(log_path, 'a') as file:
        file.write(s)
        
def plot_accuracy_epoch(accuracy_per_epoch, save_path=None, kernel_len=5):
    kernel = np.ones(kernel_len) / kernel_len
    epochs = np.arange(1, len(accuracy_per_epoch) + 1)
    
    # Trazar el gráfico de accuracy por época
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy_per_epoch, marker='o', linestyle='-')
    plt.title('Accuracy por Época')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    
    # Calcular la tendencia (trendline) mediante convolución
    trend = convolve(accuracy_per_epoch, kernel, mode='same')
    trend[:kernel_len//2] = accuracy_per_epoch[:kernel_len//2]
    trend[-kernel_len//2:] = accuracy_per_epoch[-kernel_len//2:]
    plt.plot(epochs, trend, "r--", label="Tendencia")
    
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
