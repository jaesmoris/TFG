import numpy as np
import trimesh
import scipy
import os
import matplotlib.pyplot as plt
import pickle

from scipy.interpolate import griddata
from scipy.special import sph_harm


def cartesian_coordinates_3d(r, phi, theta):
    """ Cartesian coordinates 
    Args:
        r (np.array): _description_
        phi (np.array): azimutal angle [0,2*PI]
        theta (np.array): cenital angle [0, PI]

    Returns:
        x, y, z: cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)
    return x, y, z

def spherical_coordinates(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+z**2), y)
    phi = np.arctan2(z, x)
    
    theta[theta < 0] += 2*np.pi
    phi[phi < 0] += 2*np.pi
    
    return r, phi, theta

def cartesian_coordinates_3d_2(r, phi, theta):
    """ Cartesian coordinates 
    Args:
        r (np.array): _description_
        phi (np.array): azimutal angle [0,2*PI]
        theta (np.array): cenital angle [0, PI]

    Returns:
        x, y, z: cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def spherical_coordinates_2(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    phi = np.arctan2(y, x)
    
    theta[theta < 0] += 2*np.pi
    phi[phi < 0] += 2*np.pi
    
    return r, phi, theta

def plot_harmonic_point_cloud(r, phi, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    spherical = np.column_stack((r,phi,theta))
    spherical_pos = spherical[spherical[:,0] >= 0]
    spherical_neg = spherical[spherical[:,0] < 0]
    # Extrae las coordenadas x, y, z de los puntos
    x_pos, y_pos, z_pos = cartesian_coordinates_3d(spherical_pos[:,0], spherical_pos[:,1], spherical_pos[:,2])
    x_neg, y_neg, z_neg = cartesian_coordinates_3d(np.abs(spherical_neg[:,0]), spherical_neg[:,1], spherical_neg[:,2])
    
    # Crea el gráfico 3D
    ax.scatter(x_pos, y_pos, z_pos, c='r', marker='o')
    ax.scatter(x_neg, y_neg, z_neg, c='b', marker='o')

    # Etiqueta de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Muestra el gráfico
    plt.show()

def real_spherical_harmonic(m, l, phi, theta):
    assert np.abs(m) <= l
    if m == 0:
        return np.real(scipy.special.sph_harm(m, l, phi, theta))
    elif m > 0:
        return np.real((1/np.sqrt(2))*(scipy.special.sph_harm(m, l, phi, theta)
                        + ((-1)**m) * scipy.special.sph_harm(-m, l, phi, theta)))
    elif m < 0:
        return np.real((1/(1j*np.sqrt(2)))*(scipy.special.sph_harm(-m, l, phi, theta)
                            - ((-1)**(-m)) * scipy.special.sph_harm(m, l, phi, theta)))

def simpson_integral_grids(x_grid, y_grid, eval_grid):
    n = int(x_grid.shape[0]/2)
    m = int(y_grid.shape[1]/2)

    h = (x_grid[0,-1] - x_grid[0,0]) / (2*n)    
    hx = (y_grid[-1,0] - y_grid[0,0]) / (2*m)

    J1 = np.sum(eval_grid[[0,2*n], [0,-1]]) + 2*np.sum(eval_grid[[0,2*n], 2:2*m:2]) + 4*np.sum(eval_grid[[0,2*n], 1:2*m:2])
    J2 = np.sum(eval_grid[2:2*n:2, [0,-1]]) + 2*np.sum(eval_grid[2:2*n:2, 2:2*m:2]) + 4*np.sum(eval_grid[2:2*n:2, 1:2*m:2])
    J3 = np.sum(eval_grid[1:2*n:2, [0,-1]]) + 2*np.sum(eval_grid[1:2*n:2, 2:2*m:2]) + 4*np.sum(eval_grid[1:2*n:2, 1:2*m:2])

    return (J1 + 2*J2 + 4*J3)*h*hx/9 

def simpson_integral_call(func, a, b, c, d, n, m):
    h = (b - a) / (2*n)
    J1 = 0  # Terminos extremos
    J2 = 0  # Terminos pares
    J3 = 0  # Terminos impares
    for i in range(2*n + 1):  # i in [0,2n]
        x = a + i*h
        hx = (d - c) / (2*m)
        K1 = func(x, c) + func(x, d)
        K2 = 0
        K3 = 0
        for j in range(1, 2*m):  # j in [1,2m-1]
            y = c + j*hx
            Q = func(x, y)
            if j % 2 == 0:
                K2 += Q
            else:
                K3 += Q
        L = (K1 + 2*K2 + 4*K3)*hx/3
        if i == 0 or i == 2*n:
            J1 += L
        elif i % 2 == 0:
            J2 += L
        else:
            J3 += L
    J = h*(J1 + 2*J2 + 4*J3)/3
    return J

def print_coefficients(coefficients):
    print("Coeficients (m,l)")
    l_max = int(np.sqrt(len(coefficients)))
    cont = 0
    longitud_maxima_coeficientes = max(len("{:.2f}".format(x)) for x in coefficients)
    longitud_maxima_valores = max(len(f"({m},{l})") for l in range(0, max_l) for m in range(-l, l + 1))
    for l in range(0, max_l + 1):
        for m in range(-l, l + 1):
            print("{:>{}}:".format(f"({m},{l})", longitud_maxima_valores), end=" ")
            print("{:>{}}".format("{:.2f}".format(coefficients[cont]), longitud_maxima_coeficientes), end=", ")
            cont += 1
        print("")

def compute_grids(r, phi, theta, shape = (201,201)):
    n_phi, n_theta = shape
    p = np.linspace(0, 2*np.pi, n_phi)
    t = np.linspace(0, np.pi, n_theta)

    phi_grid, theta_grid = np.meshgrid(p, t) # phi over cols, theta over rows
    # griddata((points), values, (Points at which to interpolate data))
    r_grid = griddata((phi, theta), r, (phi_grid, theta_grid), method='nearest', fill_value=0.0)
    return r_grid, phi_grid, theta_grid

def coefficient(m, l, phi_grid, theta_grid, r_grid):
    eval_grid = r_grid * np.sin(theta_grid) * real_spherical_harmonic(m, l, phi_grid, theta_grid)
    return simpson_integral_grids(phi_grid, theta_grid, eval_grid)

def spherical_coefficients(max_l, phi_grid, theta_grid, r_grid):
    coefficients = []
    for l in range(0, max_l + 1):
        for m in range(-l, l + 1):
            coefficients.append(coefficient(m, l, phi_grid, theta_grid, r_grid))
    return coefficients

def compute_coefficients_from_mesh(mesh, max_l = 30, shape = (201, 201)):
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    
    r, phi, theta = spherical_coordinates(x, y, z)
    r_grid, phi_grid, theta_grid = compute_grids(r, phi, theta, shape = shape)

    coefs = spherical_coefficients(max_l, phi_grid, theta_grid, r_grid)
    return coefs

def compute_coefficients_from_mesh2(mesh, max_l = 30, shape = (201, 201)):
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    
    r, phi, theta = spherical_coordinates_2(x, y, z)
    r_grid, phi_grid, theta_grid = compute_grids(r, phi, theta, shape = shape)

    coefs = spherical_coefficients(max_l, phi_grid, theta_grid, r_grid)
    return coefs

def spherical_harmonics_coefficients_dataset(dataset_path, destination_path, max_l = 30, shape = (201, 201)):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        # Load, compute and export
        if path[-4:] == ".stl":
            if not os.path.exists(export_path + ".pkl"):
                mesh = trimesh.load(import_path)
                coefficients = compute_coefficients_from_mesh(mesh, max_l = max_l, shape = (201, 201))
                os.makedirs(destination_path, exist_ok=True)
                with open(export_path + ".pkl", "wb") as archivo:
                    pickle.dump(coefficients, archivo)
        # Recursion over directories
        elif os.path.isdir(import_path):
            spherical_harmonics_coefficients_dataset(import_path, export_path, max_l = max_l, shape = shape)

def spherical_harmonics_coefficients_dataset2(dataset_path, destination_path, max_l = 30, shape = (201, 201)):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        # Load, compute and export
        if path[-4:] == ".stl":
            if not os.path.exists(export_path + ".pkl"):
                mesh = trimesh.load(import_path)
                coefficients = compute_coefficients_from_mesh2(mesh, max_l = max_l, shape = (201, 201))
                os.makedirs(destination_path, exist_ok=True)
                with open(export_path + ".pkl", "wb") as archivo:
                    pickle.dump(coefficients, archivo)
        # Recursion over directories
        elif os.path.isdir(import_path):
            spherical_harmonics_coefficients_dataset2(import_path, export_path, max_l = max_l, shape = shape)

def tesselation_into_grain(tesselation_mesh, phi, theta, coefficients):
    r_new = np.zeros(tesselation_mesh.vertices.shape[0])
    max_l = int(np.rint(np.sqrt(len(coefficients))))
    print(f"max_l = {max_l}")
    i = 0
    for l in range(max_l):
        for m in range(-l, l + 1):
            if not np.isnan(coefficients[i]):
                r_new += coefficients[i] * real_spherical_harmonic(m, l, phi, theta)
            i += 1
            
    xx, yy, zz = cartesian_coordinates_3d(r_new, phi, theta)
    tesselation_mesh.vertices = np.column_stack((xx, yy, zz))
    return tesselation_mesh

def tesselation_into_grain2(tesselation_mesh, phi, theta, coefficients):
    r_new = np.zeros(tesselation_mesh.vertices.shape[0])
    max_l = int(np.rint(np.sqrt(len(coefficients))))
    i = 0
    for l in range(max_l):
        for m in range(-l, l + 1):
            if not np.isnan(coefficients[i]):
                r_new += coefficients[i] * real_spherical_harmonic(m, l, phi, theta)
            i += 1
            
    xx, yy, zz = cartesian_coordinates_3d_2(r_new, phi, theta)
    tesselation_mesh.vertices = np.column_stack((xx, yy, zz))
    return tesselation_mesh

def tesselation_into_dataset(dataset_path, destination_path, tesselation_path):
    print(f"Entrando en la carpeta {dataset_path}")
    paths = os.listdir(dataset_path)
    paths.sort()
    
    tesselation_mesh = trimesh.load(tesselation_path)
    # Recontruct the grain over the spherical tesselation
    x = tesselation_mesh.vertices[:, 0]
    y = tesselation_mesh.vertices[:, 1]
    z = tesselation_mesh.vertices[:, 2]

    r, phi, theta = spherical_coordinates_2(x, y, z)
    
    
    for path in paths:
        import_path = dataset_path + "/" + path
        export_path = destination_path + "/" + path
        print(path)
        
        # Recursion over directories
        if os.path.isdir(import_path):
            tesselation_into_dataset(import_path, export_path, tesselation_path)

        # Load, compute and export
        elif not os.path.exists(export_path[:-4]):
            os.makedirs(destination_path, exist_ok=True)
            
            with open(import_path, 'rb') as archivo:
                coefficients = pickle.load(archivo)
                #coefficients = coefficients[0]
            
            mesh = tesselation_into_grain2(tesselation_mesh, phi, theta, coefficients)
            mesh.export(export_path[:-4])

    