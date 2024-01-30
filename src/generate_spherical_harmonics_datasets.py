from utils.spherical_harmonics import *


# /home/msiau/workspace/asdf/src
# /home/msiau/data/tmp/jesmoris/
dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided"
export_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided_"
shape = (201, 201)

max_l = 50
export_path_L = export_path + f"SH_L{max_l}_xyz"
spherical_harmonics_coefficients_dataset2(dataset_path, export_path_L, max_l=max_l, shape=shape)
