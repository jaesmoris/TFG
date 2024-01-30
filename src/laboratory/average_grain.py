import os
import trimesh
import numpy as np

def folder_contains_grains(folder_path):
    paths = os.listdir(folder_path)
    paths.sort()
    contains_grains = False
    for path in paths:
        if path[-4:] == ".stl":
            contains_grains = True
            break
    return contains_grains

def generate_list_of_meshes(folder_path):
    mesh_list = []
    paths = os.listdir(folder_path)
    paths.sort()
    for path in paths:
        mesh = trimesh.load(folder_path + "/" + path)
        mesh_list.append(mesh)
    return mesh_list

def generate_average_grain_from_list(mesh_list):
    verts = np.zeros(mesh_list[0].vertices.shape)
    for m in mesh_list:
        verts += m.vertices
    verts /= len(mesh_list)
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_list[0].faces)
    return mesh
    
def generate_average_grain_dataset(dataset_path, destination_path):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        if path.find("test") != -1 or path.find("val") != -1:
            continue
        
        if folder_contains_grains(import_path):
            mesh_list = generate_list_of_meshes(import_path)
            average_mesh = generate_average_grain_from_list(mesh_list)
            os.makedirs(destination_path, exist_ok=True)
            print(export_path)
            average_mesh.export(destination_path + ".stl")
            
        elif os.path.isdir(import_path):
            generate_average_grain_dataset(import_path, export_path)

dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided_20K"
destination_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided_20K_NAG"

generate_average_grain_dataset(dataset_path, destination_path)

