from utils.spherical_harmonics import *
import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh



save_path = "/home/msiau/data/tmp/jesmoris/sh_meshes"

for l in range(7):
    for m in range(-l, l+1):
        path = save_path + f"/l{l}m{m}.stl"
        mesh = spherical_harmonic_mesh(l, m, radius=5, resolution=200)
        mesh.export(path)

