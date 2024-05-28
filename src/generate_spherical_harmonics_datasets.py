from utils.spherical_harmonics import *


# /home/msiau/workspace/asdf/src
# /home/msiau/data/tmp/jesmoris/
dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided"
shape = (201, 201)

max_l = 85
spherical_harmonics_coefficients_dataset(dataset_path, max_l=max_l, shape=shape, threading_enabled=True)
