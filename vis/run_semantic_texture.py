import copy
from imageio import imread
import matplotlib
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from vis import pseudo_color_map, texture_mesh, texture_mesh_by_vertices, pseudo_colod_pointcloud_mesh, pointcloud_from_sparse_depth
from linemesh import LineMesh

def regular_512_1024():
    x = np.linspace(-2,513,32)
    y = np.linspace(-2,513,32)
    xx, yy = np.meshgrid(x, y)
    vertices = np.concatenate((xx[..., np.newaxis],yy[..., np.newaxis]),axis=-1)
    vertices = np.reshape(vertices,(-1,2))
    tri = Delaunay(vertices)
    faces = tri.simplices
    return vertices,faces

semantic_2D_img_path = "visualizations/journal/cambridge_10_0186_2D_sem.png"
save_semantic_flat_mesh_path = "visualizations/journal/cambridge_10_0186_sem_flat.obj"
init_mesh_path = "visualizations/journal/cambridge_10_0186_init.obj"
save_init_mesh_sem_path = "visualizations/journal/cambridge_10_0186_init_sem.obj"
save_init_mesh_vertex_sem_path = "visualizations/journal/cambridge_10_0186_init_vert_sem.obj"


depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
mean_depth = 70
sphere_r = 0.8
mesh_range_offset = 5
display_offset = 15

if __name__ == "__main__":

    # Step 1: Get a flat semantic mesh
    vertices,faces = regular_512_1024()    
    vertices_2D = (vertices-image_size/2)/(image_size/2*focal_length)
    vertices = np.hstack((vertices_2D,np.ones((vertices.shape[0],1))))
    vertices *= mean_depth+mesh_range_offset
    semantic_mesh_flat = o3d.geometry.TriangleMesh()
    semantic_mesh_flat.vertices = o3d.utility.Vector3dVector(vertices)
    semantic_mesh_flat.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(save_semantic_flat_mesh_path,semantic_mesh_flat)
    semantic_mesh_flat = o3d.io.read_triangle_mesh(save_semantic_flat_mesh_path)    
    texture_map = imread(semantic_2D_img_path)
    textured_mesh_flat = texture_mesh(texture_map, semantic_mesh_flat, focal_length)
    textured_mesh_flat.translate([0,0,display_offset])

    # Step 2: Load the initial mesh
    init_mesh = o3d.io.read_triangle_mesh(init_mesh_path)
    init_mesh_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(init_mesh)
    init_mesh_v = np.asarray(init_mesh.vertices)
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
    s.compute_vertex_normals()
    for i, p in enumerate(init_mesh.vertices):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        #TODO: paint semantic color
        img_idx = p[0:2]/(p[2]+0.8)*focal_length*image_size/2 + image_size/2
        if (img_idx[0] < 0 or img_idx[0] > image_size-1 or img_idx[1] < 0 or img_idx[1] > image_size-1):
            continue
        color = texture_map[np.round(img_idx[1]).astype(int),np.round(img_idx[0]).astype(int),:]
        si.paint_uniform_color(color/255)
        spheres += si    
    line_mesh1 = LineMesh(np.asarray(init_mesh_lineset.points), np.asarray(init_mesh_lineset.lines), np.ones((np.asarray(init_mesh_lineset.lines).shape[0],3))*0.4, radius=0.15)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    textured_mesh = texture_mesh(texture_map, init_mesh, focal_length)

    #o3d.io.write_triangle_mesh(save_init_mesh_sem_path,textured_mesh)
    #o3d.io.write_triangle_mesh(save_init_mesh_vertex_sem_path,spheres)
    o3d.visualization.draw_geometries([textured_mesh_flat,*line_mesh1_geoms,spheres],mesh_show_back_face=True)
    #o3d.visualization.draw_geometries([textured_mesh,spheres],mesh_show_back_face=True)
    