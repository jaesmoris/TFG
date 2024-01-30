import os
import torch
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

class CustomDatasetFromPickle(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.classes = os.listdir(data_dir)
        self.classes.sort()
        self.data = []
        self.labels = []
        
        # For every directory in the dataset
        for class_index, class_path in enumerate(self.classes):
            in_class_paths = os.listdir(data_dir + "/" + class_path)
            # For every file in the class directory
            for path in in_class_paths:
                pkl_path = data_dir + "/" + class_path + "/" + path
                with open(pkl_path, 'rb') as pkl_file:
                    pkl_object = pickle.load(pkl_file)
                self.data.append(pkl_object)
                self.labels.append(class_index)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        print(f"Dataset imported successfully! Classes: {len(self.classes)}, Samples: {self.data.shape[0]}, Sample length: {self.data.shape[1]}.")
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class CustomDatasetFromArray(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class CustomDatasetFromFold(Dataset):
    def __init__(self, data_dir, fold):
        self.data_dir = data_dir
        # Listing classes
        self.classes = os.listdir(data_dir)
        self.classes.sort()
        self.data = []
        self.labels = []
        
        # dataset/class/fold
        # For every directory matching "fold"
        for class_index, class_path in enumerate(self.classes):
            in_class_paths = os.listdir(data_dir + "/" + class_path + "/" + fold)
            in_class_paths.sort()
            # For every file in the class directory
            for path in in_class_paths:
                pkl_path = data_dir + "/" + class_path + "/" + fold + "/" + path
                with open(pkl_path, 'rb') as pkl_file:
                    pkl_object = pickle.load(pkl_file)
                self.data.append(pkl_object)
                self.labels.append(class_index)

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        print(f"Dataset imported successfully! Classes: {len(self.classes)}, Samples: {self.data.shape[0]}, Sample length: {self.data.shape[1]}.")
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

