import numpy as np
import trimesh
import os
import sys
sys.path.append('/home/msiau/workspace/asdf/src')

from utils.spherical_harmonics import *
import pickle



grain_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L50/Orkney 2ROW Scottish/OHBT16P6 10L13.stl.pkl"

destination_path = f"/home/msiau/data/tmp/jesmoris/grainss"

for i in [5, 6, 7, 8]:
    tesselation_path = f"/home/msiau/data/tmp/jesmoris/tesselations/grain_tesselation_{i}K.stl"
    tesselation_mesh = trimesh.load(tesselation_path)

    # Recontruct the grain over the spherical tesselation
    x = tesselation_mesh.vertices[:, 0]
    y = tesselation_mesh.vertices[:, 1]
    z = tesselation_mesh.vertices[:, 2]

    r, phi, theta = spherical_coordinates(x, y, z)
    with open(grain_path, 'rb') as archivo:
        coefficients = pickle.load(archivo)

    mesh = tesselation_into_grain(tesselation_mesh, phi, theta, coefficients)
    mesh.export(destination_path + f"/OHBT16P6 10L13_{i}K.stl")
