import numpy as np
import trimesh
import os
from utils.spherical_harmonics import *



dataset_path = "/home/msiau/data/tmp/jesmoris/spherical_coefficients_L50"

destination_path = f"/home/msiau/data/tmp/jesmoris/OrientedMatched_40K_L50_MOVE"
tesselation_path = f"/home/msiau/data/tmp/jesmoris/tesselations/grain_tesselation_40K.stl"
tesselation_into_dataset(dataset_path, destination_path, tesselation_path)