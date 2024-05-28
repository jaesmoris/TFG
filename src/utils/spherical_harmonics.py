import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import trimesh
from copy import deepcopy
from math import floor, ceil, sqrt, log10

from scipy.interpolate import griddata
from scipy.special import sph_harm

from tqdm.notebook import tnrange, tqdm
from ipywidgets import widgets, Layout
from IPython.display import display, clear_output
from threading import Thread

# --------- Coordinate systems ---------
def cartesian_coordinates_3d(r, phi, theta):
    """
    Convert spherical coordinates (r, phi, theta) to Cartesian coordinates (x, y, z).

    Parameters:
        r (np.array): Array of radial distances.
        phi (np.array): Array of azimuthal angles in the range [0, 2*pi].
        theta (np.array): Array of zenith angles in the range [0, pi].

    Returns:
        np.array: Array of x-coordinates.
        np.array: Array of y-coordinates.
        np.array: Array of z-coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def spherical_coordinates(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, phi, theta).

    Parameters:
        x (np.array): Array of x-coordinates.
        y (np.array): Array of y-coordinates.
        z (np.array): Array of z-coordinates.

    Returns:
        np.array: Array of radial distances.
        np.array: Array of azimuthal angles in the range [0, 2*pi].
        np.array: Array of zenith angles in the range [0, pi].
    """
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    phi = np.arctan2(y, x)
    
    theta[theta < 0] += 2*np.pi
    phi[phi < 0] += 2*np.pi
    
    return r, phi, theta

def spherical_coordinates_from_mesh(mesh):
    """
    Convert mesh vertices to spherical coordinates.

    Parameters:
        mesh (trimesh.base.Trimesh): Mesh object.

    Returns:
        np.array: Array of radial distances.
        np.array: Array of azimuthal angles in the range [0, 2*pi].
        np.array: Array of zenith angles in the range [0, pi].
    """
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    
    return spherical_coordinates(x, y, z)

def update_vertices_with_spherical_coordinates(r, phi, theta, mesh):
    """
    Update mesh vertices with new Cartesian coordinates computed from spherical coordinates.

    Parameters:
        r (np.array): Array of radial distances.
        phi (np.array): Array of azimuthal angles in the range [0, 2*pi].
        theta (np.array): Array of zenith angles in the range [0, pi].
        mesh (trimesh.base.Trimesh): Mesh object.

    Returns:
        None
    """
    x, y, z = cartesian_coordinates_3d(r, phi, theta)
    mesh.vertices = np.column_stack((x, y, z))

# --------- Spherical harmonic computation ---------
def real_spherical_harmonic(m, l, phi, theta):
    """
    Compute the real spherical harmonic function for given degree and order.

    Parameters:
        m (int): Order of the spherical harmonic.
        l (int): Degree of the spherical harmonic.
        phi (np.array): Array of azimuthal angles in the range [0, 2*pi].
        theta (np.array): Array of zenith angles in the range [0, pi].

    Returns:
        np.array: Array of real spherical harmonic values.

    Notes:
        - This function computes real spherical harmonic values from scipy.special.sph_harm function of
        the scipy module, which returns complex spherical harmonics. Real harmonics don't have a
        function in the scipy module but admits a representation using complex harmonics.
        Also, NaN values may be produced for l values above 85.
    """
    assert np.abs(m) <= l
    if m == 0:
        return np.real(sph_harm(m, l, phi, theta))
    elif m > 0:
        return np.real((1/np.sqrt(2))*(sph_harm(m, l, phi, theta)
                        + ((-1)**m) * sph_harm(-m, l, phi, theta)))
    elif m < 0:
        return np.real((1/(1j*np.sqrt(2)))*(sph_harm(-m, l, phi, theta)
                            - ((-1)**(-m)) * sph_harm(m, l, phi, theta)))

def simpson_integral_grids(x_grid, y_grid, eval_grid):
    """
    Compute the Composite Simpson's rule numerical integration over 2D grids.

    Parameters:
        x_grid (np.array): Grid of x-coordinates.
        y_grid (np.array): Grid of y-coordinates.
        eval_grid (np.array): Grid of values to be integrated.

    Returns:
        float: Integral value.

    Notes:
        - This algorithm needs to split the domain into an even number of intervals,
        so the grid shape dimensions mush be odd.
    
        - Error associated with this method:
        n, m: number of subintervals along x and y respectively.
        [a, b], [c, d]: intervals of integration along x and y respectively.
        h, k: steps of integration (b-a)/n and (d-c)/m respectively.
        f function to be integrated.
        Then, error = [(b-a)*(d-c)/180] * [h^4*(d^4f/dx^4 (t,tt)) + k^4*(d^4f/dy^4 (p,pp))]
        for some points (t,tt) and (p,pp) in [a,b]x[c,d], assuming d^4f/dx^4 and d^4f/dy^4
        are continuous.
    """
    # Number of subintervals
    n = int(x_grid.shape[0]/2)
    m = int(y_grid.shape[1]/2)

    # Horizontal and vertical steps
    h = (x_grid[0,-1] - x_grid[0,0]) / (2*n)    
    hx = (y_grid[-1,0] - y_grid[0,0]) / (2*m)

    # Edges
    J1 = (np.sum(eval_grid[[0,2*n], [0,-1]]) +
        2*np.sum(eval_grid[[0,2*n], 2:2*m:2]) +
        4*np.sum(eval_grid[[0,2*n], 1:2*m:2]))
    
    # Even indices
    J2 = (np.sum(eval_grid[2:2*n:2, [0,-1]]) +
        2*np.sum(eval_grid[2:2*n:2, 2:2*m:2]) +
        4*np.sum(eval_grid[2:2*n:2, 1:2*m:2]))
    
    # Odd indices
    J3 = (np.sum(eval_grid[1:2*n:2, [0,-1]]) +
        2*np.sum(eval_grid[1:2*n:2, 2:2*m:2]) +
        4*np.sum(eval_grid[1:2*n:2, 1:2*m:2]))

    return (J1 + 2*J2 + 4*J3)*h*hx/9

def compute_grids(r, phi, theta, shape=(201,201)):
    """
    Compute grids of radial, azimuthal, and zenith coordinates.

    Parameters:
        r (np.array): Array of radial distances.
        phi (np.array): Array of azimuthal angles in the range [0, 2*pi].
        theta (np.array): Array of zenith angles in the range [0, pi].
        shape (tuple, optional): Shape of the resulting grids. Default is (201, 201).

    Returns:
        np.array: Grid of radial distances.
        np.array: Grid of azimuthal angles.
        np.array: Grid of zenith angles.

    Notes:
        - The method used to interpolate r in a grid is to get the nearest value.
    """
    # Compute the values in the grid
    n_phi, n_theta = shape
    p = np.linspace(0, 2*np.pi, n_phi)
    t = np.linspace(0, np.pi, n_theta)

    # Compute phi and theta grids
    phi_grid, theta_grid = np.meshgrid(p, t) # phi over cols, theta over rows

    # Interpolate r grid: griddata((original points), original values, (Points at which to interpolate data))
    r_grid = griddata((phi, theta), r, (phi_grid, theta_grid), method='nearest', fill_value=0.0)
    return r_grid, phi_grid, theta_grid

def coefficient(m, l, phi_grid, theta_grid, r_grid):
    """
    Compute the coefficient of a spherical harmonic.

    Parameters:
        m (int): Order of the spherical harmonic.
        l (int): Degree of the spherical harmonic.
        phi_grid (np.array): Grid of azimuthal angles.
        theta_grid (np.array): Grid of zenith angles.
        r_grid (np.array): Grid of radial distances.

    Returns:
        float: Coefficient of the spherical harmonic.
    """
    eval_grid = r_grid * np.sin(theta_grid) * real_spherical_harmonic(m, l, phi_grid, theta_grid)
    return simpson_integral_grids(phi_grid, theta_grid, eval_grid)

def spherical_coefficients(max_l, phi_grid, theta_grid, r_grid, progress_bar=False):
    """
    Compute the coefficients of spherical harmonics decomposition.

    Parameters:
        max_l (int): Maximum degree of the spherical harmonics.
        phi_grid (np.array): Grid of azimuthal angles.
        theta_grid (np.array): Grid of zenith angles.
        r_grid (np.array): Grid of radial distances.
        progress_bar (bool, optional): If True, display a progress bar. Default is False.

    Returns:
        list: List of coefficients for each spherical harmonic.
    """
    coefficients = []
    indices = range((max_l+1)**2)
    if progress_bar:
        indices = tqdm(indices, dynamic_ncols=True)
    for index in indices:
        l = floor(sqrt(index))
        m = index - l - l**2
        coefficients.append(coefficient(m, l, phi_grid, theta_grid, r_grid))
    return coefficients

def compute_coefficients_from_mesh(mesh, max_l, shape=(201, 201), progress_bar=False):
    """
    Compute the coefficients of spherical harmonics decomposition up to
    the specified maximum degree `max_l` from a given mesh using numerical
    integration over a grid of spherical coordinates.

    Parameters:
        mesh (trimesh.base.Trimesh): Mesh object.
        max_l (int): Maximum degree of the spherical harmonics.
        shape (tuple, optional): Shape of the resulting grids. Default is (201, 201).
        progress_bar (bool, optional): If True, display a progress bar. Default is False.

    Returns:
        list: List of coefficients for each spherical harmonic.
    """
    # Spherical coordinates and grids
    r, phi, theta = spherical_coordinates_from_mesh(mesh)
    r_grid, phi_grid, theta_grid = compute_grids(r, phi, theta, shape=shape)

    # Compute coefficients
    coefficients = spherical_coefficients(max_l, phi_grid, theta_grid, r_grid, progress_bar=progress_bar)
    return coefficients

def save_coefficients_from_mesh(max_l, destination_path, path_to_element, export_path_without_extension, shape=(201, 201), progress_bar=False):
    """
    Computes and exports SH coefficients to a given path.
    """
    os.makedirs(destination_path, exist_ok=True)
    mesh = trimesh.load(path_to_element)
    print(f"Starting {path_to_element}")
    coefficients = compute_coefficients_from_mesh(mesh, max_l=max_l, shape=shape, progress_bar=progress_bar)
    print(f"Finishing {path_to_element}")

    with open(export_path_without_extension + ".pkl", "wb") as pickle_file:
        pickle.dump(coefficients, pickle_file)
    
def spherical_harmonic_mesh(l, m, radius=5, resolution=50):
    """
    Generate a mesh representing a specific spherical harmonic.

    Parameters:
        m (int): Order of the spherical harmonic.
        l (int): Degree of the spherical harmonic.
        radius (float): Radius of the spherical mesh. Default is 5.
        resolution (int): Resolution of the spherical mesh. Default is 50.

    Returns:
        trimesh.base.Trimesh: Mesh representing the specified spherical harmonic.

    Notes:
        - Negative values of the spherical harmonic cannot be represented as a radius, so the resulting
        mesh is computed from the absolut value of the radius.
    """
    # Spherical mesh to use as a base
    sphere_mesh = trimesh.creation.uv_sphere(radius=radius, count=[resolution, resolution])
    
    # Transformation into spherical coordinates
    _, phi, theta = spherical_coordinates_from_mesh(sphere_mesh)
    
    # Compute new radius as the absolut value of the harmonic
    r_new = np.zeros(phi.shape)
    r_new += np.abs(real_spherical_harmonic(m, l, phi, theta))
    
    # Recompose the new radius into cartesian coordinates        
    update_vertices_with_spherical_coordinates(r_new, phi, theta, sphere_mesh)
    
    return sphere_mesh

def spherical_harmonics_coefficients_dataset(dataset_path, max_l, destination_path=None, shape=(201, 201), threading_enabled=False, threads=None):
    """
    Computes SH decomposition for a given dataset.

    Parameters:
        dataset_path (str): Path to the folder containing the dataset.
        max_l (int): Maximum degree of the spherical harmonics.
        destination_path (str, optional): Path to the folder where the reconstructed
        dataset will be saved. If None, a default name will be set.
        shape (tuple, optional): Shape of the integration grid. Default is (201, 201).
        threading_enabled (bool, optional): If True splits the computation into threads.
        threads (list): List to store current threads.
    
    Returns:
        None
    
    Notes:
        - Items that are neither stl files nor directories will be ignored, so they won't appear in the
        new dataset.
    """
    # Default name for the new dataset
    if not destination_path:
        destination_path = f"{dataset_path}_SH_L{max_l}"

    # List of elements in folder
    elements = os.listdir(dataset_path)
    elements.sort()
    
    # List to store threads
    if threading_enabled:
        if threads is None:
            threads = []
    
    for element in elements:
        # Updating paths to clone original folder hierarchy
        path_to_element = f"{dataset_path}/{element}"
        path_to_export = f"{destination_path}/{element}"
        export_path_without_extension, extension = os.path.splitext(path_to_export)
               
        # Load, decompose and export mesh if not computed before
        if extension == ".stl" and not os.path.exists(export_path_without_extension + ".pkl"):
            # Compute and export list of coefficients
            if threading_enabled:
                t = Thread(target=save_coefficients_from_mesh,
                        args=(max_l, destination_path, path_to_element, export_path_without_extension, shape, False))
                t.start()
                threads.append(t)
            else:
                save_coefficients_from_mesh(max_l, destination_path, path_to_element, export_path_without_extension,
                                            shape=shape, progress_bar=False)
        
        # Recursion over directories
        elif os.path.isdir(path_to_element):
            spherical_harmonics_coefficients_dataset(path_to_element, max_l, destination_path=path_to_export,
                                                     shape=shape, threading_enabled=threading_enabled, threads=threads)
    
    # Wait until all threads in current folder finish
    if threading_enabled:
        for t in threads:
            t.join()

def print_coefficients(coefficients, decimals=3):
    """
    Prints coefficients of a spherical harmonics decomposition in
    a formatted manner with their corresponding degree and order.

    Parameters:
        coefficients (list): List of coefficients for each spherical harmonic.
        decimals (int, optional): Number of decimals to display. Defualt is 3.
        
    Returns:
        None
    """
    max_l = round(sqrt(len(coefficients))) - 1
    parameter_identation = floor(log10(max_l)) + 2
    template = f"(%{parameter_identation-1}d,%{parameter_identation}d): %+.{decimals}f, "
    cont = 0
    print("Coefficients (l, m)")
    for l in range(0, max_l + 1):
        list_of_strings = []
        for m in range(-l, l + 1):
            list_of_strings.append(template % (l, m, coefficients[cont]))
            cont += 1
        print("".join(list_of_strings))

def path_widget(folder_path):
    """
    Creates a text widget for displaying the current folder path. Useful for
    recursive tours through a dataset.

    Parameters:
        folder_path (str): Folder path.

    Returns:
        widget.Text: Text widget for displaying the current folder path.
    """
    widget = widgets.Text(value=folder_path, description="Current path:", layout=Layout(width='75%'))
    def update_path(change):
        clear_output(wait=True)
        display(widget)
    widget.observe(update_path, names='value')
    return widget

# --------- Grain reconstruction ---------
def tesselation_into_grain(coefficients, tesselation, phi=None, theta=None, destination_path=None, export_path=None, progress_bar=False):
    """
    Reconstructs a mesh representing a grain using spherical harmonics coefficients.

    Parameters:
        coefficients (list): Coefficients of the spherical harmonic expansion.
        tesselation (str | trimesh.base.Trimesh): If str, it represents the path to the mesh file.
            If trimesh.base.Trimesh, it represents the mesh object itself.
        phi (np.array, optional): Phi values of the tesselation mesh. If not provided, they are
            computed from the mesh vertices. Default is None.
        theta (np.array, optional): Theta values of the tesselation mesh. If not provided, they are
            computed from the mesh vertices. Default is None
        destination_path (str, optional): Folder where will be saved. Default is None. 
        export_path (str, optional): Path to export mesh. Default is None.
        progress_bar (bool, optional): If True, display a progress bar. Default is False.

    Returns:
        trimesh.base.Trimesh: Reconstructed mesh.
    """
    # Checking if tesselation is a path or a mesh
    if isinstance(tesselation, str):
        tesselation_mesh = trimesh.load(tesselation)
    elif isinstance(tesselation, trimesh.base.Trimesh):
        tesselation_mesh = deepcopy(tesselation)
    
    # Compute phi and theta values if not given
    if phi is None or theta is None:
        _, phi, theta = spherical_coordinates_from_mesh(tesselation_mesh)
    
    # Accumulate values in r_new
    r_new = np.zeros(phi.shape)
    max_l = round(sqrt(len(coefficients))) - 1
    indices = range((max_l+1)**2)
    
    # Progress bar to show the remaining time
    if progress_bar:
        indices = tqdm(indices, dynamic_ncols=True)
    
    # Iterate over coefficients and add the result on the new radius r_new
    for index in indices:
        l = floor(sqrt(index))
        m = index - l - l**2
        if not np.isnan(coefficients[index]):
            r_new += coefficients[index] * real_spherical_harmonic(m, l, phi, theta)
        else:
            print(f"Reaching NaN values evaluating scipy.special.sph_harm at l={l}, m={m}")
    
    # Recompose the new radius into cartesian coordinates        
    update_vertices_with_spherical_coordinates(r_new, phi, theta, tesselation_mesh)
    if destination_path is not None and export_path is not None:
        os.makedirs(destination_path, exist_ok=True)
        tesselation_mesh.export(export_path)
    return tesselation_mesh

def tesselation_into_dataset(dataset_path, tesselation_path, threading_enabled=False, threads=None, destination_path=None):
    """
    Reconstructs a whole dataset using a precomputed tesselation.

    Parameters:
        dataset_path (str): Path to the folder containing the dataset.
        tesselation_path (str): Path to the precomputed tesselation mesh.
        destination_path (str, optional): Path to the folder where the reconstructed dataset
            will be saved. If None, a defualt name will be set.
        threading_enables (bool, optional): If True splits the computation into threads.
        threads (list): List to store current threads.

    Returns:
        None

    Notes:
        - Items that are neither stl files nor directories will be ignored, so they won't appear in the
        new dataset.
    """
    # Default name for the new dataset
    if not destination_path:
        destination_path = f"{dataset_path}_RECONSTRUCTED"
    
    # List of elements in folder
    elements = os.listdir(dataset_path)
    elements.sort()
    
    # Compute the values ​​of phi and theta to avoid recalculating them on each call to tesselation_into_grain
    tesselation_mesh = trimesh.load(tesselation_path)
    _, phi, theta = spherical_coordinates_from_mesh(tesselation_mesh)
    
    # List to store threads
    if threading_enabled:
        if threads is None:
            threads = []

    # For every element in the current folder
    for element in elements:
        # Updating paths to clone the original folder hierarchy
        path_to_element = f"{dataset_path}/{element}"
        path_to_export = f"{destination_path}/{element}"
        export_path_without_extension, extension = os.path.splitext(path_to_export)
            
        # Load, recompose and export mesh if not computed before
        if extension == ".pkl" and not os.path.exists(export_path_without_extension + ".stl"):
            # Load coefficients
            with open(path_to_element, 'rb') as pickle_file:
                coefficients = pickle.load(pickle_file)

            # Compute and export new mesh
            if threading_enabled:
                t = Thread(target=tesselation_into_grain,
                        args=(coefficients, tesselation_mesh, phi, theta, destination_path,
                                export_path_without_extension + ".stl", False))
                t.start()
                threads.append(t)
            else:
                tesselation_into_grain(coefficients, tesselation_mesh, phi, theta, destination_path,
                             export_path_without_extension + ".stl", False)
        
        # Recursion over directories
        elif os.path.isdir(path_to_element):
            tesselation_into_dataset(path_to_element, tesselation_path, threading_enabled=threading_enabled,
                                     threads=threads, destination_path=path_to_export)
    
    # Wait until all threads in current folder finish
    if threading_enabled:
        for t in threads:
            t.join()

# --------- Average grains ---------
def generate_average_grain_from_list(mesh_list):
    """
    Generates an average grain mesh from a list of mesh objects or file paths.

    Parameters:
        mesh_list (list): List of meshes or file paths.

    Returns:
        trimesh.base.Trimesh: Average grain mesh.
    """
    # Checking if the list contains strings
    if isinstance(mesh_list[0], str):
        mesh_list = [trimesh.load(path) for path in mesh_list]
    # Define a vertices array
    vertices = np.zeros(mesh_list[0].vertices.shape)
    # For every mesh in the list
    for mesh in mesh_list:
        # Accumulate vertices
        vertices += mesh.vertices
    # Averaging vertices
    vertices /= len(mesh_list)
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh_list[0].faces)
    return mesh
    
def generate_path_list(dataset_path, path_list):
    """
    Recursively generates a list of file paths for all STL files in a directory.

    Parameters:
        dataset_path (str): Path to the directory containing the dataset.
        path_list (list): List to which the file paths will be appended.

    Returns:
        None
    """
    # List of elements in folder
    elements = os.listdir(dataset_path)
    elements.sort()
    
    # For every element in the current folder
    for element in elements:
        path_to_element = f"{dataset_path}/{element}"
        extension = os.path.splitext(path_to_element)[1]

        # Load, recompose and export mesh
        if extension == ".stl":
            path_list.append(path_to_element)

        # Recursion over directories
        elif os.path.isdir(path_to_element):
            generate_path_list(path_to_element, path_list)