import numpy as np
import copy
import trimesh


######################## FUNCTIONS ########################

def compute_middle_point(i00, i11, vertices, dict_indices):
    """Given two vertex indices, computes the middle point
    and inserts it in the index dictionary if it's not there.
    Args:
        i00 (int): index
        i11 (int): index
        vertices (np.array[x,3]): array containing all vertices
        dict_indices (dict): given an index, returns a vertex
    Returns:
        vertices (np.array[x,3]): array containing all vertices and the new one
        i01 (int): index of the new vertex 
    """
    # Keys of the dictionary always sorted
    if i00 < i11:
        i0 = i00
        i1 = i11
    else:
        i1 = i00
        i0 = i11
    
    # Check if the key exists. Get the index if so
    if (i0, i1) in dict_indices.keys():
        i01 = dict_indices[(i0, i1)]
    else:
        # If not compute the middle point
        p0 = vertices[i0, :]
        p1 = vertices[i1, :]
        p01 = (p0 + p1) / 2
        p01 = p01 / np.linalg.norm(p01) # Point normalization
        vertices = np.vstack((vertices, p01))
        i01 = vertices.shape[0] - 1
        dict_indices[(i0, i1)] = i01
    return vertices, i01
def split_triangular_face(iface, faces, vertices, dict_indices):
    # Get the indices of the face vertexs
    face = faces[iface]
    i0 = face[0]
    i1 = face[1]
    i2 = face[2]
    # Compute the middle points and their indices
    vertices, i01 = compute_middle_point(face[0], face[1], vertices, dict_indices)
    vertices, i02 = compute_middle_point(face[0], face[2], vertices, dict_indices)
    vertices, i12 = compute_middle_point(face[1], face[2], vertices, dict_indices)
    
    # Compute the new faces
    faces = np.vstack((faces, [np.sort([i0, i01, i02])]))
    faces = np.vstack((faces, [np.sort([i1, i01, i12])]))
    faces = np.vstack((faces, [np.sort([i2, i02, i12])]))
    faces = np.vstack((faces, [np.sort([i01, i02, i12])]))
    
    # Delete olf face
    faces = np.delete(faces, iface, 0)
    
    return faces, vertices
def split_triangular_fractal(faces, vertices, dict_indices):
    n_faces = faces.shape[0]
    for i in range(n_faces):
        faces, vertices = split_triangular_face(0, faces, vertices, dict_indices)
    return faces, vertices
def split_square_face(iface, faces, vertices, dict_indices):
    # Get the indices of the face vertexs
    face = faces[iface]
    i0 = face[0]
    i1 = face[1]
    i2 = face[2]
    i3 = face[3]
    
    # Compute the middle points and their indices
    vertices, i01 = compute_middle_point(face[0], face[1], vertices, dict_indices)
    vertices, i12 = compute_middle_point(face[1], face[2], vertices, dict_indices)
    vertices, i23 = compute_middle_point(face[2], face[3], vertices, dict_indices)
    vertices, i30 = compute_middle_point(face[3], face[0], vertices, dict_indices)
    vertices, i0123 = compute_middle_point(i01, i23, vertices, dict_indices)
    
    # Compute the new faces
    faces = np.vstack((faces, np.array([i0, i01, i0123, i30])))
    faces = np.vstack((faces, np.array([i01, i1, i12, i0123])))
    faces = np.vstack((faces, np.array([i0123, i12, i2, i23])))
    faces = np.vstack((faces, np.array([i30, i0123, i23, i3])))
    
    # Delete olf face
    faces = np.delete(faces, iface, 0)
    
    return faces, vertices
def split_square_fractal(faces, vertices, dict_indices):
    n_faces = faces.shape[0]
    for i in range(n_faces):
        faces, vertices = split_square_face(0, faces, vertices, dict_indices)
    return faces, vertices
def split_square_into_triangles(faces, vertices, dict_indices):
    n_faces = faces.shape[0]
    triangular_faces = []
    for i in range(n_faces):
        face = faces[i, :]
        triangular_faces.append([face[0], face[1], face[2]])
        triangular_faces.append([face[0], face[2], face[3]])
    return np.array(triangular_faces)
        
def fix_faces(faces, vertices):
    for i in range(faces.shape[0]):
        face = faces[i]
        u = vertices[face[1]] - vertices[face[0]]
        v = vertices[face[2]] - vertices[face[0]]
        w = np.cross(u, v)
        if np.inner(w, vertices[face[0]]) < 0:
            faces[i] = [face[0], face[2], face[1]]
    return faces

######################## OCTAHEDRON ########################

vertices_octahedron = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
faces_octahedron = np.array([[0, 1, 2],
                 [0, 1, 5],
                 [0, 4, 5],
                 [0, 2, 4],
                 [3, 4, 5],
                 [1, 2, 3],
                 [2, 3, 4],
                 [1, 3, 5]])

######################## ICOSAHEDRON ########################

X = .525731112119133606;
Z = .850650808352039932;
N = 0;

vertices_icosahedron = np.array([
    [-X,N,Z], [X,N,Z], [-X,N,-Z], [X,N,-Z],
    [N,Z,X], [N,Z,-X], [N,-Z,X], [N,-Z,-X],
    [Z,X,N], [-Z,X, N], [Z,-X,N], [-Z,-X, N]
])
faces_icosahedron = np.array([
    [0, 1, 4],
    [0, 4, 9],
    [9, 4, 5],
    [4, 8, 5],
    [4, 1, 8],
    [8, 1, 10],
    [8, 10, 3],
    [5, 8, 3],
    [5, 3, 2],
    [2, 3, 7],
    [7, 3, 10],
    [7, 10, 6],
    [7, 6, 11],
    [11, 6, 0],
    [0, 6, 1],
    [6, 10, 1],
    [9, 11, 0],
    [9, 2, 11],
    [9, 5, 2],
    [7, 11, 2]
])

# DEPTH:
# 4 --> V=2562      C=5120      <1MB
# 5 --> V=10242     C=20480     1MB
# 6 --> V=40962     C=81920     4MB
# 7 --> V=163842    C=327680    16MB

######################## HEXAHEDRON ########################
#       0--------1      
#      /        /|          4---5
#     /        / |          | D |
#    /        /  |      4---0---1---5---4
#   3--------2   |      | C | A | E | F |
#   |   4    |   5      7---3---2---6---7
#   |        |  /           | B |
#   |        | /            7---6
#   |        |/
#   7--------6

sq = 1.0/np.sqrt(3)
vertices_hexahedron = np.array([
                    [-sq, sq, -sq], # (-x,+y,-z) -> 0
                    [sq, sq, -sq],  # (+x,+y,-z) -> 1
                    [sq, sq, sq],   # (+x,+y,+z) -> 2
                    [-sq, sq, sq],  # (-x,+y,+z) -> 3
                    [-sq, -sq, -sq],# (-x,-y,-z) -> 4
                    [sq, -sq, -sq], # (+x,-y,-z) -> 5
                    [sq, -sq, sq],  # (+x,-y,+z) -> 6
                    [-sq, -sq, sq]])# (-x,-y,+z) -> 7
faces_hexahedron = np.array([
                [0, 1, 2, 3],   # A
                [3, 2, 6, 7],   # B
                [4, 0, 3, 7],   # C
                [4, 5, 1, 0],   # D
                [1, 5, 6, 2],   # E
                [5, 4, 7, 6]])  # F

dict_indices = {}


faces = faces_hexahedron
vertices = vertices_hexahedron

depth = 5
for i in range(depth):
    faces, vertices = split_square_fractal(faces, vertices, dict_indices)

t_faces = split_square_into_triangles(faces, vertices, dict_indices)
t_faces = fix_faces(t_faces, vertices)
# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=vertices, faces=t_faces)
mesh.export(f"/home/msiau/data/tmp/jesmoris/tesselations/hexahedron{depth}.stl")

