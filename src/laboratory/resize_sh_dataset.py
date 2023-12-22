import pickle
import numpy as np
import os
import sys
sys.path.append('/home/msiau/workspace/asdf/src')
from utils.spherical_harmonics import *

max_l = 80
last_index = (max_l + 1)**2
dataset_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L100"
destination_path = f"/home/msiau/data/tmp/jesmoris/spherical_coefficients_L{max_l}"

def resize_dataset_sh(dataset_path, destination_path):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        # Load, compute and export
        if path[-8:] == ".stl.pkl":
            if not os.path.exists(export_path + ".pkl"):
                with open(import_path, "rb") as archivo:
                    coefficients = pickle.load(archivo)
                
                os.makedirs(destination_path, exist_ok=True)
                with open(export_path, "wb") as archivo:
                    pickle.dump(coefficients[:last_index], archivo)
        # Recursion over directories
        elif os.path.isdir(import_path):
            resize_dataset_sh(import_path, export_path)
            
resize_dataset_sh(dataset_path, destination_path)

