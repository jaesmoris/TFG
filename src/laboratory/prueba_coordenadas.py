# Prueba que demuestra que no importa el orden de las coordenadas xyz o xzy

import numpy as np
import trimesh
import scipy
import os
import matplotlib.pyplot as plt
import pickle
from copy import copy
from scipy.interpolate import griddata
from scipy.special import sph_harm
from utils.spherical_harmonics import *


max_l = 30

mesh_path = "/home/msiau/data/tmp/jesmoris/Oriented_grains/Dundee 6ROW Scandinavian/DHBT16P89 4.stl"
mesh = trimesh.load(mesh_path)
tesselation_path = "/home/msiau/data/tmp/jesmoris/tesselation4.stl"
tesselation_mesh = trimesh.load(tesselation_path)

x = tesselation_mesh.vertices[:, 0]
y = tesselation_mesh.vertices[:, 1]
z = tesselation_mesh.vertices[:, 2]

r1, phi1, theta1 = spherical_coordinates(x, y, z)
r2, phi2, theta2 = spherical_coordinates_2(x, y, z)

coefs1 = compute_coefficients_from_mesh(mesh, max_l)
coefs2 = compute_coefficients_from_mesh2(mesh, max_l)
    
r_new_1 = np.zeros(tesselation_mesh.vertices.shape[0])
r_new_2 = np.zeros(tesselation_mesh.vertices.shape[0])

mesh1 = tesselation_into_grain(tesselation_mesh, phi1, theta1, coefs1)
mesh1.export("./mesh1.stl")

mesh2 = tesselation_into_grain2(tesselation_mesh, phi2, theta2, coefs2)
mesh2.export("./mesh2.stl")