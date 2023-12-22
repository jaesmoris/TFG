from utils.spherical_harmonics import *


# /home/msiau/workspace/asdf/src
# /home/msiau/data/tmp/jesmoris/
dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_grains"
export_path = "/home/msiau/data/tmp/jesmoris/"
shape = (201, 201)

max_l = 80
export_path_L = export_path + f"spherical_coefficients_L{max_l}_xyz"
spherical_harmonics_coefficients_dataset2(dataset_path, export_path_L, max_l=max_l, shape=shape)
