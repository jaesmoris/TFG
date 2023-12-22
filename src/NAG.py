import os
import trimesh
import numpy as np


def grain_dist(g1, g2):
    mse = (g1.vertices - g2.vertices)**2
    return np.sum(np.sum(mse))

def grain_dist2(g1, g2):
    mse = (g1.vertices - g2.vertices)**2
    mse = np.sqrt(np.sum(mse, axis=1))
    return np.sum(mse)

mesh_path = "/home/msiau/data/tmp/jesmoris/reconstructed_grains_from_coefficients_L50_T6/Dundee 6ROW Faro/DHBT16T93 4.stl"
mesh = trimesh.load(mesh_path)

average_grains_folder_path = "/home/msiau/workspace/asdf/models/nearest_average_grain/all_labels"
average_grains_paths = os.listdir(average_grains_folder_path)
average_grains_paths.sort()


dists = []

for average_grains_path in average_grains_paths:
    avg_mesh = trimesh.load(average_grains_folder_path + "/" + average_grains_path)
    dists.append(grain_dist(avg_mesh, mesh))

dists = np.array(dists)
index = np.argmin(dists)
print(dists)
print("La clase es: " + average_grains_paths[index])
    
