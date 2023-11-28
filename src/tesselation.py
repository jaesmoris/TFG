import numpy as np
import copy
import trimesh

def compute_middle_point(i00, i11, vertices, dict_indices):
    if i00 < i11:
        i0 = i00
        i1 = i11
    else:
        i1 = i00
        i0 = i11
    
    p0 = vertices[i0, :]
    p1 = vertices[i1, :]
        
    if (i0, i1) in dict_indices.keys():
        i01 = dict_indices[(i0, i1)]
    else:
        p01 = (p0 + p1) / 2
        p01 = p01 / np.linalg.norm(p01)
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
    
def epoch(faces, vertices, dict_indices):
    n_faces = faces.shape[0]
    for i in range(n_faces):
        faces, vertices = split_triangular_face(0, faces, vertices, dict_indices)
    return faces, vertices

def fix_faces(faces, vertices):
    for i in range(faces.shape[0]):
        face = faces[i]
        u = vertices[face[1]] - vertices[face[0]]
        v = vertices[face[2]] - vertices[face[0]]
        w = np.cross(u, v)
        if np.inner(w, vertices[face[0]]) < 0:
            faces[i] = [face[0], face[2], face[1]]
    return faces

vertices_8 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])

faces_8 = np.array([[0, 1, 2],
                 [0, 1, 5],
                 [0, 4, 5],
                 [0, 2, 4],
                 [3, 4, 5],
                 [1, 2, 3],
                 [2, 3, 4],
                 [1, 3, 5]])

X = .525731112119133606;
Z = .850650808352039932;
N = 0;

vertices = np.array([
    [-X,N,Z], [X,N,Z], [-X,N,-Z], [X,N,-Z],
    [N,Z,X], [N,Z,-X], [N,-Z,X], [N,-Z,-X],
    [Z,X,N], [-Z,X, N], [Z,-X,N], [-Z,-X, N]
])

faces = np.array([
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

dict_indices = {}

'''faces, vertices = epoch(faces, vertices, dict_indices)
'''
faces, vertices = epoch(faces, vertices, dict_indices)
faces, vertices = epoch(faces, vertices, dict_indices)
faces, vertices = epoch(faces, vertices, dict_indices)
faces, vertices = epoch(faces, vertices, dict_indices)
faces, vertices = epoch(faces, vertices, dict_indices)
#faces, vertices = epoch(faces, vertices, dict_indices)





faces = fix_faces(faces, vertices)


# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export("../data/icosaedron.stl")

'''print(vertices)
print(faces)
print(dict_indices)'''
'''arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(arr)
print(np.delete(arr, 0, 0))
print(arr)'''
#print(mesh.face_normals)
