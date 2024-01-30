import os
import trimesh
import numpy as np
import pickle

def folder_contains_grains(folder_path):
    paths = os.listdir(folder_path)
    paths.sort()
    contains_grains = False
    for path in paths:
        if path[-4:] == ".stl":
            contains_grains = True
            break
    return contains_grains

def generate_list_of_meshes(folder_path):
    mesh_list = []
    paths = os.listdir(folder_path)
    paths.sort()
    for path in paths:
        mesh = trimesh.load(folder_path + "/" + path)
        mesh_list.append(mesh)
    return mesh_list

def generate_average_grain_from_list(mesh_list):
    verts = np.zeros(mesh_list[0].vertices.shape)
    for m in mesh_list:
        verts += m.vertices
    verts /= len(mesh_list)
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh_list[0].faces)
    return mesh

def grain_dist(g1, g2):
    mse = (g1.vertices - g2.vertices)**2
    return np.sum(np.sum(mse))

def grain_dist2(g1, g2):
    mse = (g1.vertices - g2.vertices)**2
    mse = np.sqrt(np.sum(mse, axis=1))
    return np.sum(mse)

def get_test_meshes(experiment_folder_path):
    classes = os.listdir(experiment_folder_path)
    classes.sort()
    mesh_list = []
    labels = []
    for index, label in enumerate(classes):
        mesh_paths = os.listdir(f"{experiment_folder_path}/{label}/test")
        mesh_paths.sort()
        for mesh_path in mesh_paths:
            path = f"{experiment_folder_path}/{label}/test/{mesh_path}"
            mesh_list.append(trimesh.load(path))
            labels.append(index)
    return mesh_list, labels, classes

def get_test_meshes_pkl(experiment_folder_path):
    classes = os.listdir(experiment_folder_path)
    classes.sort()
    pkl_list = []
    labels = []
    for index, label in enumerate(classes):
        mesh_paths = os.listdir(f"{experiment_folder_path}/{label}/test")
        mesh_paths.sort()
        for mesh_path in mesh_paths:
            path = f"{experiment_folder_path}/{label}/test/{mesh_path}"
            with open(path, 'rb') as pkl_file:
                pkl_object = pickle.load(pkl_file)
            pkl_list.append(pkl_object)
            labels.append(index)
    print(len(labels))
    return pkl_list, labels, classes

def predict(mesh, class_meshes):
    dists = [grain_dist(grain, mesh) for grain in class_meshes]
    dists = np.array(dists)
    index = np.argmin(dists)
    return index

def test_all(test_meshes, labels, class_meshes):
    confusion_matrix = np.zeros(shape=(len(class_meshes),len(class_meshes)), dtype=np.int_)
    for i in range(len(labels)):
        pred = predict(test_meshes[i], class_meshes)
        confusion_matrix[labels[i], pred] += 1
    return confusion_matrix
'''
experiment = "Bere"
experiment_folder = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_20K/{experiment}"
average_grains_folder_path = f"/home/msiau/data/tmp/jesmoris/Oriented_Divided_20K_NAG/{experiment}"
average_grains_paths = os.listdir(average_grains_folder_path)
average_grains_paths.sort()


class_meshes = generate_list_of_meshes(average_grains_folder_path)
test_meshes, labels, classes  = get_test_meshes(experiment_folder)
print(len(labels))
#print(classes)

cm = test_all(test_meshes, labels, class_meshes)
print("\nConfusion matrix:")
print(cm)
ncm = cm.astype(np.float64)
acc = 0
for i in range(len(np.unique(labels))):
    ncm[i] /= np.sum(ncm[i])
    acc += cm[i,i]
acc /= np.sum(np.sum(cm))
print("\nNormalized confusion matrix:")
print(ncm)
print("\nAccuracy:")
print(acc)
for i, cl in enumerate(classes):
    print(f"{i}. {cl}")
'''
