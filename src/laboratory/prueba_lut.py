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


max_l = 100
total_coefs = (max_l+1)**2

pkl_path = f"/home/msiau/data/tmp/jesmoris/spherical_coefficients_L{max_l}/Dundee 2ROW British/DHBT16P16 1.stl.pkl"
with open(pkl_path, 'rb') as archivo:
    coefficients = pickle.load(archivo)

tesselation_path = "/home/msiau/data/tmp/jesmoris/tesselation4.stl"
tesselation_mesh = trimesh.load(tesselation_path)

x = tesselation_mesh.vertices[:, 0]
y = tesselation_mesh.vertices[:, 1]
z = tesselation_mesh.vertices[:, 2]

r, phi, theta = spherical_coordinates(x, y, z)
    
r_new_forward = np.zeros(tesselation_mesh.vertices.shape[0])
r_new_backward = np.zeros(tesselation_mesh.vertices.shape[0])

i = 0
lut_coefficients = {}
for l in range(max_l + 1):
    for m in range(-l, l + 1):
        lut_coefficients[i] = [m, l]
        i += 1
print("lut completada")

indices_forward = [j for j in range(total_coefs)]
indices_backward = copy(indices_forward)
indices_backward.reverse()
#print(indices_backward)



for k in indices_forward:
    lut_c = lut_coefficients[k]
    if not np.isnan(coefficients[k]):
        r_new_forward += coefficients[k] * real_spherical_harmonic(lut_c[0], lut_c[1], phi, theta)
    #print(k, lut_c, coefficients[k], r_new_forward[:10])
print(r_new_forward[:10])

for k in indices_backward:
    lut_c = lut_coefficients[k]
    if not np.isnan(coefficients[k]):
        r_new_backward += coefficients[k] * real_spherical_harmonic(lut_c[0], lut_c[1], phi, theta)
print(r_new_backward[:10])

