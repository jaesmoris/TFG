import os
import trimesh
import numpy as np

dataset_path = "/home/msiau/data/tmp/jesmoris/reconstructed_grains_from_coefficients_L50_T6"
export_path = "/home/msiau/data/tmp/jesmoris/average_grain.stl"

def generate_old_school_dataset(dataset_path, mesh_list):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        
        # Load mesh
        if path[-4:] == ".stl":
            mesh = trimesh.load(import_path)
            mesh_list.append(mesh)

        # Recursion over directories
        elif os.path.isdir(import_path):
            generate_old_school_dataset(import_path, mesh_list)


mesh_list = []

generate_old_school_dataset(dataset_path, mesh_list)
print(f"Importados {len(mesh_list)} granos.")

verts = np.zeros(mesh_list[0].vertices.shape)
for m in mesh_list:
    verts += m.vertices

verts /= len(mesh_list)

mesh_list[0].vertices = verts
mesh_list[0].export(export_path)

