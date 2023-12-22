import numpy as np
import trimesh
import os
import pickle as pkl
from utils.spherical_harmonics import *


# Paths
tesselation_path = "/home/msiau/data/tmp/jesmoris/tesselation6.stl"
export_path = "/home/msiau/data/tmp/jesmoris/DHBT16P89 7.stl"
pkl_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L20/Dundee 6ROW Scandinavian/DHBT16P89 7.stl.pkl"



# Load meshes
tesselation_mesh = trimesh.load(tesselation_path)
with open(pkl_path, 'rb') as archivo:
    coefficients = pickle.load(archivo)
    coefficients = coefficients[0]

# Recontruct the grain over the spherical tesselation
x = tesselation_mesh.vertices[:, 0]
y = tesselation_mesh.vertices[:, 1]
z = tesselation_mesh.vertices[:, 2]

r_new = np.zeros(tesselation_mesh.vertices.shape[0])
r, phi, theta = spherical_coordinates(x, y, z)

max_l = int(np.rint(np.sqrt(len(coefficients))))
i = 0

print(f"shape of r_new: {r_new.shape}")
print(f"shape of phi: {phi.shape}")
print(f"len coefficients: {len(coefficients)}")
print(f" coefficients[5]: {coefficients[5]}")

for l in range(max_l):
    for m in range(-l, l + 1):
        r_new += coefficients[i] * real_spherical_harmonic(m, l, phi, theta)
        i += 1
        
xx, yy, zz = cartesian_coordinates_3d(r_new, phi, theta)
tesselation_mesh.vertices = np.column_stack((xx, yy, zz))
tesselation_mesh.export(export_path)
