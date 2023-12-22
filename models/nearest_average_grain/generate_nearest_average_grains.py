import numpy as np
import trimesh
import os
import sys
sys.path.append('/home/msiau/workspace/asdf/src')

from utils.classification import *

# Paths
dataset_path = "/home/msiau/data/tmp/jesmoris/reconstructed_grains_from_coefficients_L50_T6"
export_dataset_path = "/home/msiau/workspace/asdf/models/nearest_average_grain/all_labels"
tesselation_path = "/home/msiau/data/tmp/jesmoris/tesselations/tesselation6.stl"
tesselation_mesh = trimesh.load(tesselation_path)


class_paths = os.listdir(dataset_path)
class_paths.sort()
print(class_paths)
# For each class
for i, class_path in enumerate(class_paths):
    average_vertices = np.zeros(tesselation_mesh.vertices.shape)
    grain_paths = os.listdir(dataset_path + "/" + class_path)
    grain_paths.sort()
    # For each grain
    for j, grain_path in enumerate(grain_paths):
        # Load mesh
        path = dataset_path + "/" + class_path + "/" + grain_path
        mesh = trimesh.load(path)
        # Accumulate vertices
        average_vertices += mesh.vertices
    average_vertices /= len(grain_paths)
    tesselation_mesh.vertices = average_vertices
    tesselation_mesh.export(export_dataset_path + "/" + class_path + ".stl")
