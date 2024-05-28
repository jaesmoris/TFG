import os
import pickle
import trimesh
from utils.slices import *
from utils.orientaciones import *

def generate_old_school_dataset(dataset_path, destination_path):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        # Load, compute and export
        if path[-4:] == ".stl":
            if not os.path.exists(export_path + ".pkl"):
                mesh = trimesh.load(import_path)
                descriptors_list = sample_slices(mesh)
                os.makedirs(destination_path, exist_ok=True)
                with open(export_path + ".pkl", "wb") as archivo:
                    pickle.dump(descriptors_list, archivo)
        # Recursion over directories
        elif os.path.isdir(import_path):
            generate_old_school_dataset(import_path, export_path)


dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided"
export_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided_old_school"

generate_old_school_dataset(dataset_path, export_path)
