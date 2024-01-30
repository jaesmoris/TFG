import numpy as np
import os
import trimesh

from tesselation import (
    split_square_face,
    split_square_fractal,
    split_square_into_triangles,
    fix_faces,
    faces_hexahedron,
    vertices_hexahedron,
    sq)

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

mesh_path = "/home/msiau/workspace/asdf/src/sq1.stl"
mesh = trimesh.load(mesh_path)

shape = (, )
radius_img = np.zeros(())


