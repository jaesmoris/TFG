import numpy as np
import trimesh
from utils.spherical_harmonics import *
from time import time

n_phi, n_theta = (201, 201)
p = np.linspace(0, 2*np.pi, n_phi)
t = np.linspace(0, np.pi, n_theta)

phi_grid, theta_grid = np.meshgrid(p, t)

eval_grid = np.array([[i*np.sqrt(j) for i in range(n_phi)] for j in range(n_theta)])
t0 = time()
for i in range(100):
    num1 = simpson_integral_grids(phi_grid, theta_grid, eval_grid)
t1 = time()
for i in range(100):
    num2 = simpson_integral_grids_222(phi_grid, theta_grid, eval_grid)
t2 = time()
print(f"Temps vell: {t1-t0}")
print(f"Temps nou: {t2-t1}")
print(f"Speedup: {(t1-t0)/(t2-t1)}")
print(f"Són iguals: {num1 == num2}")
print(f"num1: {num1}")
print(f"num2: {num2}")
