import pickle
import numpy as np
import os
import sys
import trimesh

sys.path.append('/home/msiau/workspace/asdf/src')

def find_path(mesh_name, dataset_search_path):
    for directorio_actual, _, archivos in os.walk(dataset_search_path):
        for archivo in archivos:
            #print(archivo)
            if archivo.find(mesh_name) != -1:
                return os.path.join(directorio_actual, archivo)

    # Si no se encuentra el archivo
    print(f"NO SE HA ENCONTRADO EL GRANO {mesh_name}")
    return -1


def into_oriented_dataset(dataset_path, dataset_search_path, import_name, export_name):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = import_path.replace(import_name, export_name)
        
        # Recursion over directories
        if os.path.isdir(import_path):
            into_oriented_dataset(import_path, dataset_search_path, import_name, export_name)

        # Load, compute and export
        else:
            if os.path.exists(export_path):
                continue
            os.makedirs(dataset_path.replace(import_name, export_name), exist_ok=True)
            pathh = find_path(path[3:], dataset_search_path)
            if pathh == -1:
                print(export_path)
                continue
            mesh = trimesh.load(pathh)
            mesh.export(export_path)

# Recorremos este dataset
dataset_structure_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided"
# Buscamos en este dataset
dataset_import_path = "/home/msiau/data/tmp/jesmoris/OrientedMatched_20K_L50"
# Guardamos en este dataset
import_name = "Oriented_Divided"
export_name = "Oriented_Divided_20K"
into_oriented_dataset(dataset_structure_path, dataset_import_path, import_name, export_name)
