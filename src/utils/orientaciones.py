"Rotation functions"
import math
import copy
import numpy as np
import pyrender
import trimesh
from sklearn.decomposition import PCA
#import meshcut as mc
import matplotlib.pyplot as plt
from .slices import xsection, ysection, zsection
from .figuras_predefinidas import create_axis, render_axis


def rodrigues_rotation_matrix(omega, theta):
    """Calcula la matriz de rotación de ángulo theta
    respecto el eje omega.
    Args:
        omega (np.array | List): eje de rotación
        theta (float): ángulo de rotación
    Returns:
        np.array[3,3]: matriz de rotación
    """
    if (isinstance(omega, np.ndarray) and len(omega.shape) == 1
            and omega.shape[0] == 3):
        o_norm = omega/np.linalg.norm(omega)

    elif isinstance(omega, list) and len(omega) == 3:
        o_norm = np.array(omega)
        o_norm = omega/np.linalg.norm(omega)

    else:
        raise TypeError("El eje tiene que ser una lista o on array numpy")

    o_matrix = np.array([
        [0, -o_norm[2], o_norm[1]],
        [o_norm[2], 0, -o_norm[0]],
        [-o_norm[1], o_norm[0], 0]
    ])
    rotation_matrix = (np.identity(3) + o_matrix * np.sin(theta) +
                       np.matmul(o_matrix, o_matrix) * (1 - np.cos(theta)))
    return rotation_matrix


def x_rotation_matrix(theta):
    """Calcula la matriz de rotación antihoraria
    de ángulo theta respecto el eje x.
    Args:
        theta (float): ángulo de rotación
    Returns:
        np.array[3,3]: matriz de rotación
    """
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
        ])
    return rotation_matrix_x


def y_rotation_matrix(theta):
    """Calcula la matriz de rotación antihoraria
    de ángulo theta respecto el eje y.
    Args:
        theta (float): ángulo de rotación
    Returns:
        np.array[3,3]: matriz de rotación
    """
    rotation_matrix_y = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
        ])
    return rotation_matrix_y


def z_rotation_matrix(theta):
    """Calcula la matriz de rotación antihoraria
    de ángulo theta respecto el eje z.
    Args:
        theta (float): ángulo de rotación
    Returns:
        np.array[3,3]: matriz de rotación
    """
    rotation_matrix_z = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
        ])
    return rotation_matrix_z


def align_x(mesh):
    """Alinea el grano de forma que el vector propio
    de valor propio dominante repose sobre el eje x.
    Args:
        mesh (Trimesh.mesh): mesh del grano

    Returns:
        Trimesh.mesh: mesh del grano reorientado
    """
    # CENTERING MESH
    mesh.vertices = mesh.vertices - mesh.center_mass
    
    # Principal Component Analysis
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    cov = pca.get_covariance()
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    #print("Eigenvalues: \n", eigenvalues)
    #print("Eigenvectors: \n", eigenvectors)
    #print("Determinant eigenvectors: \n", np.linalg.det(eigenvectors))
    
    # SORTING EIGENVECTORS
    biggest_axis = 0
    ind = np.argmax(eigenvalues)
    if ind != biggest_axis:
        eigenvectors[:, [biggest_axis, ind]] = eigenvectors[:, [ind, biggest_axis]]
        eigenvalues[[biggest_axis, ind]] = eigenvalues[[ind, biggest_axis]]
        mesh.vertices[:, [biggest_axis, ind]] = mesh.vertices[:, [biggest_axis, ind]]

    smallest_axis = 2
    ind = np.argmin(eigenvalues)
    if ind != smallest_axis:
        eigenvectors[:, [smallest_axis, ind]] = eigenvectors[:, [ind, smallest_axis]]
        eigenvalues[[smallest_axis, ind]] = eigenvalues[[ind, smallest_axis]]
        mesh.vertices[:, [biggest_axis, ind]] = mesh.vertices[:, [biggest_axis, ind]]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] = -eigenvectors[:, 0]
    #print("Eigenvalues: \n", eigenvalues)
    #print("Eigenvectors: \n", eigenvectors)
    #print("Determinant eigenvectors: \n", np.linalg.det(eigenvectors))

    mesh.vertices = np.matmul(eigenvectors.transpose(), mesh.vertices.transpose()).transpose()
    return mesh


def sum_of_dist(A, B):
    dist = 0
    for v in A:
        #dist = dist + np.sum(np.sqrt(np.sum(np.square(B - v), axis=1)))
        #dist = dist + np.sum(np.sum(np.square(B - v), axis=1))
        dist += np.min(np.power(np.sum(np.square(B - v), axis=1), 3))

    return dist


def perimeter(curve):
    """Given a curve, returns the perimeter of the curve
    assuming it is sorted.
    
    Args:
        curve (np.array): Array of points with shape (x,3)
        
    Returns:
        float: The perimeter of the curve
    """
    n_vertices = curve.shape[0]
    diferences = curve[0:n_vertices-1] - curve[1:n_vertices]
    return np.sum(np.sqrt(np.sum(np.square(diferences), axis=1)))


def polar_coordinates(x, y):
    """Given x and y in numpy arrays, returns the polar coordinates.

    Args:
        x (np.array): x coordinates
        y (np.array): y coordinates

    Returns:
        np.array: returns the polar coordinates r and theta in numpy arrays.
    """
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)
    return r, theta


def cartesian_coordinates_2d(r, theta):
    """Given r and theta in numpy arrays, returns the cartesian coordinates.

    Args:
        r (np.array): radial coordinate
        theta (np.aaray): angle

    Returns:
        np.array: returns x and y cartesian coordinates in two numpy arrays.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def align_y(mesh_original):
    "Slices the mesh on X axis"
    mesh = copy.deepcopy(mesh_original)
    mesh.vertices /= np.max(np.abs(mesh.vertices[:,0]))/3.5
    xsections = [
        xsection(mesh, origin_plane=np.array([2,0,0])),
        xsection(mesh, origin_plane=np.array([1,0,0])),
        xsection(mesh, origin_plane=np.array([0,0,0])),
        xsection(mesh, origin_plane=np.array([-1,0,0])),
        xsection(mesh, origin_plane=np.array([-2,0,0]))
    ]
    correct_orientation = []
    for xsec in xsections:
        radius, theta = polar_coordinates(xsec[:,2], xsec[:,1])
        ind = np.argmin(radius)
        r_0 = radius[ind]
        t_0 = theta[ind]
        z_0, y_0 = cartesian_coordinates_2d(r_0, t_0)
        #print(z_0, y_0)
        if z_0 > 0:
            correct_orientation.append(1)
        else:
            correct_orientation.append(-1)
    correct_orientation = np.array(correct_orientation)
    if np.mean(correct_orientation) < 0:
        mesh_original.vertices = mesh_original.vertices = np.matmul(x_rotation_matrix(np.pi), mesh_original.vertices.transpose()).transpose()
    return mesh_original


def align(mesh):
    mesh = align_x(mesh)
    mesh = align_y(mesh)

    zmin = np.min(mesh.vertices[:, 2])
    zmax = np.max(mesh.vertices[:, 2])
    delta = (zmax-zmin)*0.05

    zsec = zsection(mesh, origin_plane=np.array([0, 0, zmin+delta]))
    if np.mean(zsec[:, 0]) < 0:
        mesh.vertices = mesh.vertices = np.matmul(z_rotation_matrix(np.pi), mesh.vertices.transpose()).transpose()
        #print("Cambiazo")
    return mesh


def lookAt(eye, target, up, yz_flip=False):
    # https://github.com/Jianghanxiao/Helper3D/tree/master/trimesh_render
    # Normalize the up vector
    up /= np.linalg.norm(up)
    forward = eye - target
    forward /= np.linalg.norm(forward)
    if np.dot(forward, up) == 1 or np.dot(forward, up) == -1:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    # Construct a rotation matrix from the right, new_up, and forward vectors
    rotation = np.eye(4)
    rotation[:3, :3] = np.row_stack((right, new_up, forward))

    # Apply a translation to the camera position
    translation = np.eye(4)
    translation[:3, 3] = [
        np.dot(right, eye),
        np.dot(new_up, eye),
        -np.dot(forward, eye),
    ]

    if yz_flip:
        # This is for different camera setting, like Open3D
        rotation[1, :] *= -1
        rotation[2, :] *= -1
        translation[1, 3] *= -1
        translation[2, 3] *= -1

    camera_pose = np.linalg.inv(np.matmul(translation, rotation))

    return camera_pose

def custom_scene(render_mesh, camera_eye, camera_target, camera_up, axis_flag=False):
    # ESCENA XYZ
    scene = pyrender.Scene()

    if axis_flag:
        meshx, meshy, meshz = create_axis(length=4.0, radius=0.05)
        render_meshx, render_meshy, render_meshz = render_axis(meshx, meshy, meshz)
        scene.add(render_meshx)
        scene.add(render_meshy)
        scene.add(render_meshz)

    scene.add(render_mesh)

    # CAMERA
    #camera_eye = np.array([4,4,4], dtype=np.float64)
    #camera_target = np.array([0,0,0], dtype=np.float64)
    #camera_up = np.array([0,1,0], dtype=np.float64)
    camera_pose = lookAt(camera_eye, camera_target, camera_up, yz_flip=False)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    '''# LIGHT
    eye = np.array([-5,0,0], dtype=np.float64)
    target = np.array([0,0,0], dtype=np.float64)
    up = np.array([0,1,0], dtype=np.float64)
    light_pose = lookAt(eye, target, up, yz_flip=False)'''

    light = pyrender.PointLight(intensity=50.0)
    scene.add(light, pose=camera_pose)
    return scene



def flip_and_roll(x, y):
    _, theta = polar_coordinates(x, y)
    xx = x
    yy = y
    sorting_indices = np.argsort(theta)
    sorted_theta = theta[sorting_indices]
    xx = x[sorting_indices]
    yy = y[sorting_indices]
    arg_min = np.argmin(np.abs(sorted_theta))
    xx = np.roll(xx, -arg_min)
    yy = np.roll(yy, -arg_min)
    return xx, yy



def curvatura_2d(x, y):
    # DERIVADAS
    x_d = (np.roll(x, 1) - x) * x.shape[0]
    x_dd = (np.roll(x_d, 1) - x_d) * x.shape[0]
    y_d = (np.roll(y, 1) - y) * y.shape[0]
    y_dd = (np.roll(y_d, 1) - y_d) * y.shape[0]
    
    # CURVATURA 2D
    return (x_d * y_dd - x_dd * y_d) / ((x_d**2 + y_d**2)**(3/2))


def sample_slice(sec, n_samples, coord):
    coord1 = 0
    coord2 = 0
    if coord == "x":
        coord1 = 1
        coord2 = 2
    if coord == "y":
        coord1 = 0
        coord2 = 2
    if coord == "z":
        coord1 = 0
        coord2 = 1
    sec = np.column_stack(flip_and_roll(sec[:, coord1],sec[:, coord2]))
    sec = sec[np.linspace(0, sec.shape[0], n_samples+1, dtype=np.int16)[:-1]]
    sec = sec.flatten()
    return sec

def sample_slices(mesh):
    # Number of sampled points
    n_samples = 50
    descriptors_list = [mesh.area, mesh.volume, mesh.convex_hull.volume, mesh.volume/mesh.convex_hull.volume]
    
    # Deltas
    xdelta = (np.max(mesh.vertices[:, 0]) - np.min(mesh.vertices[:, 0]))/2
    ydelta = (np.max(mesh.vertices[:, 1]) - np.min(mesh.vertices[:, 1]))/2
    #zdelta = (np.max(mesh.vertices[:, 2]) - np.min(mesh.vertices[:, 2]))/2
    
    # Weights
    xweight = 0.33
    yweight = 0.33
    #zweight = 0.33
    
    # Sections
    xsec_0 = xsection(mesh, origin_plane=[0,0,0])
    xsec_1_pos = xsection(mesh, origin_plane=[xdelta*xweight,0,0])
    xsec_1_neg = xsection(mesh, origin_plane=[-xdelta*xweight,0,0])
    xsec_2_pos = xsection(mesh, origin_plane=[xdelta*xweight*2,0,0])
    xsec_2_neg = xsection(mesh, origin_plane=[-xdelta*xweight*2,0,0])
    ysec_0 = ysection(mesh, origin_plane=[0,0,0])
    ysec_1_pos = ysection(mesh, origin_plane=[0,ydelta*yweight,0])
    ysec_1_neg = ysection(mesh, origin_plane=[0,-ydelta*yweight,0])
    zsec_0 = zsection(mesh, origin_plane=[0,0,0])
    
    # Concatenating arrays
    descriptors_list.extend(sample_slice(xsec_0, n_samples, "x"))
    descriptors_list.extend(sample_slice(xsec_1_pos, n_samples, "x"))
    descriptors_list.extend(sample_slice(xsec_1_neg, n_samples, "x"))
    descriptors_list.extend(sample_slice(xsec_2_pos, n_samples, "x"))
    descriptors_list.extend(sample_slice(xsec_2_neg, n_samples, "x"))
    descriptors_list.extend(sample_slice(ysec_0, n_samples, "y"))
    descriptors_list.extend(sample_slice(ysec_1_pos, n_samples, "y"))
    descriptors_list.extend(sample_slice(ysec_1_neg, n_samples, "y"))
    descriptors_list.extend(sample_slice(zsec_0, n_samples, "z"))
    
    return descriptors_list
