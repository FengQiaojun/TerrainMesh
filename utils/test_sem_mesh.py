## Just a quick script to render a semantic texture mesh
import numpy as np 
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.renderer import TexturesUV
import open3d as o3d
from imageio import imread
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified

input_mesh_file = "/mnt/NVMe-2TB/qiaojun/SensatUrban/birmingham_5/Meshes/0012.obj"
input_texture = "/mnt/NVMe-2TB/qiaojun/journal_terrain/visualizations/7_gt.png"
#input_texture = "/mnt/NVMe-2TB/qiaojun/SensatUrban/birmingham_5/Semantics_5/0012.png"




mesh = o3d.io.read_triangle_mesh(input_mesh_file)
vertices = np.array(mesh.vertices)
faces = np.array(mesh.triangles)
vertices_2D = vertices[:,0:2]/vertices[:,2:3] * 2
vertices_2D = (vertices_2D+1)/2

triangle_uvs = np.zeros((faces.shape[0]*3,2))
for i in range(triangle_uvs.shape[0]):
    face_id = i//3
    face_v_id = i%3
    v_id = faces[face_id,face_v_id]
    triangle_uvs[i,:] = vertices_2D[v_id,:]

texture_map = imread(input_texture)
#texture_map = convert_class_to_rgb_sensat_simplified(texture_map)

mesh.textures=[o3d.geometry.Image(texture_map)]
mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
#o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
o3d.io.write_triangle_mesh("temp.obj",mesh)