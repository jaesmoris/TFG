from sklearn.model_selection import train_test_split
from utils.spherical_harmonics import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import trimesh

def find_best_l_max(mesh_path, coefficients_path=None, save_path=None, plot_path=None, test_size=0.3):
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Get point cloud
    point_cloud = mesh.vertices
    point_cloud_train, point_cloud_test = train_test_split(point_cloud, test_size=test_size, random_state=42)
    
    # Compute coefficients
    mesh_train = trimesh.Trimesh(vertices=point_cloud_train, faces=[])
    if not os.path.exists(coefficients_path):
        coefficients = compute_coefficients_from_mesh(mesh_train, max_l=70, progress_bar=False)
        with open(coefficients_path, "wb") as pickle_file:
            pickle.dump(coefficients, pickle_file)
    else:
        with open(coefficients_path, 'rb') as coefficients_file:
            coefficients = pickle.load(coefficients_file)
    
    if save_path:
        with open(mesh_path[:-4] + ".txt", 'w') as file:
            file.write("L2 l_values for grains:\n")
    
    # list of l_max value and L2 dist
    distances_L2 = []
    l_max_list = [i for i in range(50, 70)]
    for l_max in l_max_list:
        # Reconstruct the original mesh
        n = (l_max + 1)**2
        mesh_reconstructed = tesselation_into_grain(coefficients[:n], mesh_path)

        # Split vertices into train and test
        _, point_cloud_reconstructed_test = train_test_split(mesh_reconstructed.vertices, test_size=0.3, random_state=42)

        # Compute error on test vertices
        error = point_cloud_test - point_cloud_reconstructed_test
        dist = np.sum(np.sqrt(np.sum(error * error, axis=1)))
        if save_path:
            with open(mesh_path[:-4] + ".txt", 'a') as file:
                file.write(f"{[l_max, dist]}\n")
        print([l_max, dist])
        distances_L2.append([l_max, dist])
    
    distances_L2 = np.array(distances_L2)
    
    if save_path:
        with open(save_path, "wb") as pickle_file:
            pickle.dump(distances_L2, pickle_file)
    
    if plot_path:
        name, extension = os.path.splitext(os.path.basename(mesh_path))
        plt.plot(distances_L2[:,0], distances_L2[:,1])
        plt.title(f'L2 distance vs l_max of {name}')
        plt.savefig(plot_path)
    
    optimal_l = np.argmin(distances_L2[:,1])
    return distances_L2, optimal_l
'''
    mesh_paths = [  "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P10 1.stl",
                    "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P10 6.stl",
                    "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P16 2.stl",
                    "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P4 1.stl",
                    "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P88 1.stl"]

    coefficients_paths = ["/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P10 1.pkl",
                     "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P10 6.pkl",
                     "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P16 2.pkl",
                     "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P4 1.pkl",
                     "/home/msiau/data/tmp/jesmoris/best_sh/K1_DHBT16P88 1.pkl"]
'''
root_dir = "/home/msiau/data/tmp/jesmoris/best_sh2/"

paths = ['OHBT16P6 1.stl', 'DHBT16P88 1.stl', 'OBT16P33 2.stl',
              'DHBT16P16 1.stl', 'OBT16P33 3.stl', 'OBT16P33 1.stl',
              'OHBT16P17 3.stl', 'DHBT16P10 3.stl', 'DHBT16T93 3.stl',
              'DHBT16P16 3.stl', 'OHBT16P6 2.stl', 'OHBT16P17 2.stl',
              'DHBT16T93 1.stl', 'DHBT16T93 2.stl', 'OHBT16P17 1.stl',
              'OHBT16P6 3.stl', 'DHBT16P10 1.stl', 'DHBT16P10 2.stl',
              'DHBT16P88 3.stl', 'DHBT16P88 2.stl', 'DHBT16P16 2.stl']

mesh_paths = [root_dir + path for path in paths]
coefficients_paths = [root_dir + path[:-4] + ".pkl" for path in paths]

log_path = "/home/msiau/data/tmp/jesmoris/best_sh2/values_l_max.txt"
with open(log_path, 'w') as file:
    file.write("Best l_values for grains:\n")
for mesh_path, coefficients_path in zip(mesh_paths, coefficients_paths):
    _, optimal_l = find_best_l_max(mesh_path, coefficients_path=coefficients_path,
                                   save_path="/home/msiau/data/tmp/jesmoris/best_sh2/results.pkl",
                                   plot_path="/home/msiau/data/tmp/jesmoris/best_sh2/plot_l_max.png")
    with open(log_path, 'a') as file:
        file.write(f"{mesh_path} -> {optimal_l}\n")
