import numpy as np
import trimesh
import os

from utils.spherical_harmonics import *

# Load meshes
tesselation_mesh = trimesh.load("../data/tetraedron.stl")

# Recontruct the grain over the spherical tesselation
x = tesselation_mesh.vertices[:, 0]
y = tesselation_mesh.vertices[:, 1]
z = tesselation_mesh.vertices[:, 2]

r_new = np.zeros(tesselation_mesh.vertices.shape[0])
r, phi, theta = spherical_coordinates_2(x, y, z)

max_l = 30
i = 0
export_path = "../data/harmonics/"
os.makedirs(export_path, exist_ok=True)
for l in range(max_l + 1):
    for m in range(-l, l + 1):
        r_new = real_spherical_harmonic(m, l, phi, theta)
        xx, yy, zz = cartesian_coordinates_3d_2(r_new, phi, theta)
        tesselation_mesh.vertices = np.column_stack((xx, yy, zz))
        tesselation_mesh.export(export_path + f"l{l}_m{m}.stl")
        i += 1

