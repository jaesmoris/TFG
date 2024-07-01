from sklearn.model_selection import train_test_split
from utils.spherical_harmonics import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import trimesh

def list_meshes_from_folder(root_dir):
    list_of_elements = [path for path in os.listdir(root_dir) if path[-4:] == ".stl"]
    list_of_elements.sort()
    list_of_paths = [root_dir + "/" + path for path in list_of_elements]
    return list_of_paths, list_of_elements

def load_meshes(list_of_paths):
    list_of_meshes = []
    for path in list_of_paths:
        if path[-4:] == ".stl":
            mesh = trimesh.load(path)
            list_of_meshes.append(mesh)
    return list_of_meshes

def split_point_cloud(list_of_meshes):
    train_list = []
    test_list = []
    
    # Get point cloud
    for mesh in list_of_meshes:
        point_cloud = mesh.vertices
        point_cloud_train, point_cloud_test = train_test_split(point_cloud, test_size=0.3, random_state=42)
        mesh_train = trimesh.Trimesh(vertices=point_cloud_train, faces=[])
        mesh_test = trimesh.Trimesh(vertices=point_cloud_test, faces=[])
        train_list.append(mesh_train)
        test_list.append(mesh_test)
    
    return train_list, test_list

def decomposition_SH(list_of_paths, list_of_point_clouds):
    list_of_SH_decompositions = []
    for i in range(len(list_of_paths)):
        path_to_element = list_of_paths[i]
        pkl_path = path_to_element[:-4] + ".pkl"
        if not os.path.exists(pkl_path):
            coefficients = compute_coefficients_from_mesh(list_of_point_clouds[i], max_l=85, progress_bar=False)
            with open(pkl_path, "wb") as pickle_file:
                pickle.dump(coefficients, pickle_file)
        else:
            with open(pkl_path, 'rb') as coefficients_file:
                coefficients = pickle.load(coefficients_file)
        list_of_SH_decompositions.append(coefficients)
    return list_of_SH_decompositions

def compute_MSE_error(test_mesh, reconstructed_mesh):
    error = test_mesh.vertices - reconstructed_mesh.vertices
    mse = np.sum(np.sum(error * error, axis=1)) / error.size
    return mse

def reconstruction_SH_one_grain(mesh, coefficients):
    _, phi, theta = spherical_coordinates_from_mesh(mesh)
    
    # Accumulate values in r_new
    r_new = np.zeros(phi.shape)
    max_l = 85
    index = 0
    
    errors = []
    
    for l in range(max_l + 1):
        print(f"l = {l}")
        reconstructed_mesh = deepcopy(mesh)
        for m in range(-l, l + 1):
            if not np.isnan(coefficients[index]):
                r_new += coefficients[index] * real_spherical_harmonic(m, l, phi, theta)
            else:
                print(f"Reaching NaN values evaluating scipy.special.sph_harm at l={l}, m={m}")
            index += 1
        update_vertices_with_spherical_coordinates(r_new, phi, theta, reconstructed_mesh)
        error = compute_MSE_error(mesh, reconstructed_mesh)
        errors.append(error)
    return errors

def reconstruction_SH(test_meshes, list_of_SH_decompositions):
    mse_errors = []
    for test_mesh, coefficients in zip(test_meshes, list_of_SH_decompositions):
        mse_list = reconstruction_SH_one_grain(test_mesh, coefficients)
        mse_errors.append(mse_list)
    return mse_errors

def save_results(mse_errors, list_of_elements, root_dir):
    with open(root_dir + "/results.pkl", "wb") as pkl_file:
        pickle.dump([mse_errors, list_of_elements], pkl_file)

def load_results(root_dir):
    with open(root_dir + "/results.pkl", "rb") as pkl_file:
        pkl_info = pickle.load(pkl_file)
        mse_errors = pkl_info[0]
        list_of_elements = pkl_info[1] 
    return mse_errors, list_of_elements

from matplotlib.ticker import MultipleLocator
def plot_superimposed_lists(data_lists, labels):
    """
    Esta función superpone varias listas en una gráfica de matplotlib.
    
    :param data_lists: Lista de listas, donde cada lista interna contiene 86 valores.
    :param labels: Lista de etiquetas para cada una de las listas internas.
    """
    start = 20
    end = 85
    step = 5
    
    fig, ax = plt.subplots()
    if len(data_lists) != len(labels):
        raise ValueError("El número de listas de datos y etiquetas debe ser igual.")
    
    for data, label in zip(data_lists, labels):
        if len(data) != 86:
            # raise ValueError("Cada lista de datos debe tener exactamente 86 elementos.")
            print("upss")
        y = data[start:end]
        x = np.arange(start,end)
        ax.plot(x,y, label=label[:-4])
        # Usar MaxNLocator para asegurarse de que el eje x tenga solo valores enteros
        ax.xaxis.set_major_locator(MultipleLocator(1))
        # Establecer el punto de inicio del eje x en 3
        ax.set_xlim(start, end)
        ax.set_xticks(np.arange(start, end, step))
    
    plt.xlabel('l_max')
    plt.ylabel('MSE')
    plt.title('Error aproximando por armónicos')
    plt.legend()
    plt.savefig('curvas_granos2080.png')
    plt.show()
    
def plot_average(data_lists, labels):
    """
    Esta función superpone varias listas en una gráfica de matplotlib.
    
    :param data_lists: Lista de listas, donde cada lista interna contiene 86 valores.
    :param labels: Lista de etiquetas para cada una de las listas internas.
    """
    start = 30
    end = 85
    step = 5
    
    fig, ax = plt.subplots()
    if len(data_lists) != len(labels):
        raise ValueError("El número de listas de datos y etiquetas debe ser igual.")
    
    np_data = np.array(data_lists)
    np_data = np.mean(np_data, axis=0)
    y = np_data[start:end]
    x = np.arange(start, end)
    ax.plot(x,y)
    # Usar MaxNLocator para asegurarse de que el eje x tenga solo valores enteros
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # Establecer el punto de inicio del eje x en 3
    ax.set_xlim(start, end)
    ax.set_xticks(np.arange(start, end, step))
    
    plt.xlabel('l_max')
    plt.ylabel('MSE')
    plt.title('Error aproximando por armónicos')
    plt.savefig('curvas_granos_average2080.png')
    plt.show()

def run_best_sh(root_dir):
    # list_of_paths, list_of_elements = list_meshes_from_folder(root_dir)
    # # Load meshes
    # print("Loading...")
    # meshes = load_meshes(list_of_paths)
    # # Split
    # print("Spliting into train/test...")
    # train_list, test_list = split_point_cloud(meshes)
    # # SH decomposition
    # print("SH decompositions...")
    # list_of_SH_decompositions = decomposition_SH(list_of_paths, train_list)
    # # Reconstruction and errors
    # print("SH recompositions...")
    # mse_errors = reconstruction_SH(test_list, list_of_SH_decompositions)
    # save_results(mse_errors, list_of_elements, root_dir)
    # Plot
    mse_errors, list_of_elements = load_results(root_dir)
    plot_superimposed_lists(mse_errors, list_of_elements)
    plot_average(mse_errors, list_of_elements)






if __name__ == "__main__":
    
    root_dir = "/home/msiau/data/tmp/jesmoris/best_sh33"
    
    run_best_sh(root_dir)
    