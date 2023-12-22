import numpy as np
import trimesh

from utils.spherical_harmonics import *

# Load meshes
tesselation_mesh = trimesh.load("../data/tetraedron.stl")
grain_mesh = trimesh.load("../data/Barley STL oriented files/Dundee 6ROW Scandinavian/DHBT16P101 8.stl")

# Compute spherical coefficients
max_l = 100
coefs = compute_coefficients_from_mesh(grain_mesh, max_l = max_l)

# Recontruct the grain over the spherical tesselation
x = tesselation_mesh.vertices[:, 0]
y = tesselation_mesh.vertices[:, 1]
z = tesselation_mesh.vertices[:, 2]

r_new = np.zeros(tesselation_mesh.vertices.shape[0])
r, phi, theta = spherical_coordinates(x, y, z)
i = 0
for l in range(max_l + 1):
    for m in range(-l, l + 1):
        r_new += real_spherical_harmonic(m, l, phi, theta) * coefs[i]
        i += 1
xx, yy, zz = cartesian_coordinates_3d(r_new, phi, theta)

tesselation_mesh.vertices = np.column_stack((xx, yy, zz))


export_path = "../data/tesselation/"
tesselation_mesh.export(export_path + "asdf.stl")
grain_mesh.export(export_path + "original.stl")




