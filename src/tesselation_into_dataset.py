import numpy as np
import trimesh
import os
from utils.spherical_harmonics import *



dataset_path = "/home/msiau/data/tmp/jesmoris/Oriented_Divided_SH_L50_xyz"

destination_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_10K"
tesselation_path = f"/home/msiau/data/tmp/jesmoris/tesselations/grain_tesselation_10K.stl"
tesselation_into_dataset(dataset_path, destination_path, tesselation_path)