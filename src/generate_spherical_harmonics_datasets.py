from utils.spherical_harmonics import *

dataset_path = "./data/Barley STL oriented files"
shape = (201, 201)

'''max_l = 30
export_path = f"../data/spherical_coefficients_L{max_l}"
spherical_harmonics_coefficients_dataset(dataset_path, export_path, max_l=max_l, shape=shape)

max_l = 30
export_path = f"../data/spherical_coefficients_L{max_l}"
spherical_harmonics_coefficients_dataset(dataset_path, export_path, max_l=max_l, shape=shape)

max_l = 30
export_path = f"../data/spherical_coefficients_L{max_l}"
spherical_harmonics_coefficients_dataset(dataset_path, export_path, max_l=max_l, shape=shape)'''

for max_l in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    export_path = f"./data/spherical_coefficients_L{max_l}"
    spherical_harmonics_coefficients_dataset(dataset_path, export_path, max_l=max_l, shape=shape)
