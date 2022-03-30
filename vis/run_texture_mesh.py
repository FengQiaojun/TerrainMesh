from imageio import imread
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from vis import pseudo_color_map, texture_mesh, texture_mesh_by_vertices

def regular_512_1024():
    x = np.linspace(-2,513,32)
    y = np.linspace(-2,513,32)
    xx, yy = np.meshgrid(x, y)
    vertices = np.concatenate((xx[..., np.newaxis],yy[..., np.newaxis]),axis=-1)
    vertices = np.reshape(vertices,(-1,2))
    tri = Delaunay(vertices)
    faces = tri.simplices
    return vertices,faces

mesh_path = "visualizations/journal/cambridge_10_0186_refine.obj"
rgb_img_path = "visualizations/journal/cambridge_10_0186.png"
depth_img_path = "visualizations/journal/cambridge_10_0186_refine_depth.png"
sem_img_path = "visualizations/journal/cambridge_10_0186_refine_sem.png"
save_rgb_mesh_path = "visualizations/journal/cambridge_10_0186_rgb.obj"
save_depth_mesh_path = "visualizations/journal/cambridge_10_0186_depth.obj"
save_sem_mesh_path = "visualizations/journal/cambridge_10_0186_sem.obj"
#save_rgb_flat_mesh_path = "visualizations/journal/cambridge_10_0186_rgb_flat.obj"
save_rgb_flat_mesh_path = "visualizations/journal/cambridge_10_0186_rgb_flat_pseudo.obj"

depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
mean_depth = 70

if __name__ == "__main__":

    
    '''
    depth_map = imread(depth_img_path)/100
    #print(np.min(depth_map))
    #print(np.max(depth_map*(depth_map < 600)))

    pseudo_depth_map = pseudo_color_map(depth_map,depth_min,depth_max)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    
    # RGB mesh (with RGB texture)
    texture_map = imread(rgb_img_path)
    textured_mesh = texture_mesh(texture_map, mesh, focal_length)
    o3d.visualization.draw_geometries([textured_mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(save_rgb_mesh_path,textured_mesh)

       
    # depth mesh
    texture_map = pseudo_depth_map
    textured_mesh = texture_mesh(texture_map, mesh, focal_length)
    o3d.visualization.draw_geometries([textured_mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(save_depth_mesh_path,textured_mesh)
    

    # semantic mesh
    texture_map = imread(sem_img_path)
    textured_mesh = texture_mesh(texture_map, mesh, focal_length)
    o3d.visualization.draw_geometries([textured_mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(save_sem_mesh_path,textured_mesh)
    '''