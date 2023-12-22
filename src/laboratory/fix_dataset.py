import numpy as np
import trimesh
import os
import pickle as pkl
from utils.spherical_harmonics import *

dataset_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L40"
def fix_dataset(dataset_path):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    
    for path in paths:
        import_path = dataset_path + "/" + path

        # Recursion over directories
        if os.path.isdir(import_path):
            fix_dataset(import_path)

        else:
            with open(import_path, 'rb') as archivo:
                coefficients = pickle.load(archivo)
                if isinstance(coefficients[0], list):
                    coefficients = coefficients[0]
            with open(import_path, 'wb') as archivo:
                pickle.dump(coefficients, archivo)

fix_dataset(dataset_path)