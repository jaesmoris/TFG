import numpy as np
import matplotlib.pyplot as plt
import meshcut as mc


def section(mesh, origin_plane, normal_plane):
    return np.array(mesh.section(plane_origin=origin_plane, plane_normal=normal_plane).vertices)



def xsection(mesh, origin_plane=np.array([0,0,0])):
    return section(mesh, origin_plane, np.array([1,0,0]))

def ysection(mesh, origin_plane=np.array([0,0,0])):
    return section(mesh, origin_plane, np.array([0,1,0]))

def zsection(mesh, origin_plane=np.array([0,0,0])):
    return section(mesh, origin_plane, np.array([0,0,1]))



def plot_xsection(mesh, origin_plane=np.array([0,0,0])):
    "Alzado"
    section = xsection(mesh, origin_plane=origin_plane)
    y = section[:, 1]
    z = section[:, 2]
    plt.scatter(z, y)
    plt.title(f'Slice x={origin_plane[0]}')
    plt.xlabel('Z Axis')
    plt.ylabel('Y Axis')
    plt.axis('equal')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.scatter(2, 2, color='red', s=0.1)
    plt.scatter(-2, -2, color='red', s=0.1)
    plt.scatter(-2, 2, color='red', s=0.1)
    plt.scatter(2, -2, color='red', s=0.1)
    plt.show()
    
def plot_ysection(mesh, origin_plane=np.array([0,0,0])):
    section = ysection(mesh, origin_plane=origin_plane)
    x = section[:, 0]
    z = section[:, 2]
    plt.scatter(z, x)
    plt.title(f'Slice y={origin_plane[1]}')
    plt.xlabel('Z Axis')
    plt.ylabel('X Axis')
    plt.axis('equal')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.scatter(2, 2, color='red', s=0.1)
    plt.scatter(-2, -2, color='red', s=0.1)
    plt.scatter(-2, 2, color='red', s=0.1)
    plt.scatter(2, -2, color='red', s=0.1)

    plt.show()

def plot_zsection(mesh, origin_plane=np.array([0,0,0])):
    section = zsection(mesh, origin_plane=origin_plane)
    x = section[:, 0]
    y = section[:, 1]
    plt.scatter(x, y)
    plt.title(f'Slice z={origin_plane[2]}')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.axis('equal')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.scatter(2, 2, color='red', s=0.1)
    plt.scatter(-2, -2, color='red', s=0.1)
    plt.scatter(-2, 2, color='red', s=0.1)
    plt.scatter(2, -2, color='red', s=0.1) #    return x, y
    plt.show()

'''
# get a single cross section of the mesh
slice = mesh.section(plane_origin=[0,0,1.25],
                    plane_normal=[0,0,1])
# we can move the 3D curve to a Path2D object easily
slice_2D, to_3D = slice.to_planar()
slice_2D.show()
z_extents = mesh.bounds[:,2]
# slice every .125 model units (eg, inches)
z_levels  = np.arange(*z_extents, step=.125)
sections = mesh.section_multiplane(plane_origin=mesh.bounds[0],
                                plane_normal=[0,0,1],
                                heights=z_levels)
sections
combined = np.sum(sections[12:])
combined.show()
'''