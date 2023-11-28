"Figuras predeterminadas"
import numpy as np
import pyrender
import trimesh


def create_axis_x_mesh(length=5.0, radius=0.02, num_secciones=8):
    vertices = []
    faces = []

    # Parte superior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = length
        y = radius * np.sin(angle)
        z = radius * np.cos(angle)
        vertices.append([x, y, z])

    # Parte inferior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = 0
        y = radius * np.sin(angle)
        z = radius * np.cos(angle)
        vertices.append([x, y, z])

    # Caras del cilindro
    for i in range(num_secciones):
        next_i = (i + 1) % num_secciones
        # Cara de la parte superior
        faces.append([i, next_i, i + num_secciones])
        # Cara de la parte inferior
        faces.append([i + num_secciones, next_i, next_i + num_secciones])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32))


def create_axis_y_mesh(length=5.0, radius=0.02, num_secciones=8):
    vertices = []
    faces = []

    # Crear los vértices de la parte superior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = radius * np.cos(angle)
        y = length
        z = radius * np.sin(angle)
        vertices.append([x, y, z])

    # Crear los vértices de la parte inferior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = radius * np.cos(angle)
        y = 0
        z = radius * np.sin(angle)
        vertices.append([x, y, z])

    # Crear las caras del cilindro
    for i in range(num_secciones):
        next_i = (i + 1) % num_secciones
        # Cara de la parte superior
        faces.append([i, next_i, i + num_secciones])
        # Cara de la parte inferior
        faces.append([i + num_secciones, next_i, next_i + num_secciones])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32))


def create_axis_z_mesh(length=5.0, radius=0.02, num_secciones=8):
    vertices = []
    faces = []

    # Crear los vértices de la parte superior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = length
        vertices.append([x, y, z])

    # Crear los vértices de la parte inferior del cilindro
    for i in range(num_secciones):
        angle = 2 * np.pi * i / num_secciones
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        vertices.append([x, y, z])

    # Crear las caras del cilindro
    for i in range(num_secciones):
        next_i = (i + 1) % num_secciones
        # Cara de la parte superior
        faces.append([i, next_i, i + num_secciones])
        # Cara de la parte inferior
        faces.append([i + num_secciones, next_i, next_i + num_secciones])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32))


def create_axis(length=5.0, radius=0.02):
    meshx = create_axis_x_mesh(length, radius)
    meshy = create_axis_y_mesh(length, radius)
    meshz = create_axis_z_mesh(length, radius)
    return meshx, meshy, meshz


def render_axis(meshx, meshy, meshz):
    render_meshx = pyrender.Mesh.from_trimesh(
        meshx, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 0.0, 0.0, 1.0]))  # RED
    render_meshy = pyrender.Mesh.from_trimesh(meshy, material=pyrender.MetallicRoughnessMaterial(
	baseColorFactor=[0.0, 1.0, 0.0, 1.0])) # GREEN
    render_meshz = pyrender.Mesh.from_trimesh(meshz, material=pyrender.MetallicRoughnessMaterial(
	baseColorFactor=[0.0, 0.0, 1.0, 1.0])) # BLUE
    return render_meshx, render_meshy, render_meshz


def create_centered_sphere_mesh(radius, resolution, center_point):
    # Crear una esfera centrada en el origen
    sphere_mesh = trimesh.creation.uv_sphere(radius=radius, count=[resolution, resolution])

    # Mover la esfera al punto deseado
    sphere_mesh.apply_translation(center_point)

    return sphere_mesh


def create_plane_mesh(origin, size=(1.0, 1.0), resolution=(10, 10)):
    # Create a grid of points on the plane
    u = np.linspace(-size[0] / 2, size[0] / 2, resolution[0])
    v = np.linspace(-size[1] / 2, size[1] / 2, resolution[1])
    u, v = np.meshgrid(u, v)

    # Compute the points on the plane
    points = origin + u[:, :, np.newaxis] * np.array([0, 0, 1]) + v[:, :, np.newaxis] * np.array([1, 0, 0])

    # Assign a constant height along the z-axis
    z = np.zeros_like(u)

    # Create the vertices and faces for the trimesh
    vertices = np.column_stack([points.reshape(-1, 3)])
    faces = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v1 = i * resolution[1] + j
            v2 = v1 + 1
            v3 = (i + 1) * resolution[1] + j
            v4 = v3 + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Create the trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh