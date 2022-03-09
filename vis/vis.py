import copy
from imageio import imread,imwrite
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from linemesh import LineMesh

# Convert a scalar map to a pseudo-color map
# inputs:
#   img:            2D scale map (H x W)
#   depth_min:      Min depth in the img
#   depth_max:      Max depth in the img
#   camp:           The color map. By default cm.terrain.
# outputs:
#   pseudocolor_img:    2D pseudo-color map (H x W x 3)

def pseudo_color_map(img,depth_min,depth_max,cmap=cm.terrain):
    norm = matplotlib.colors.Normalize(vmin=int(depth_min), vmax=int(depth_max), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    pseudocolor_img = mapper.to_rgba(img)
    pseudocolor_img = pseudocolor_img[:,:,:3]*255
    return pseudocolor_img.astype(np.uint8)


def pseudo_color_map_sparse(img,depth_min,depth_max,dot_r=3,cmap=cm.terrain):
    norm = matplotlib.colors.Normalize(vmin=int(depth_min), vmax=int(depth_max), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
    visible_x,visible_y = np.where(img>0)
    pseudocolor_img = np.zeros((img.shape[0],img.shape[1],3))
    pseudocolor_img[:,:,:] = [1,1,1]
    for pixel_x,pixel_y in zip(visible_x,visible_y):
        color = mapper.to_rgba(img[pixel_x,pixel_y])
        for x in range(max(0,pixel_x-dot_r),min(img.shape[0],pixel_x+dot_r+1)):
            for y in range(max(0,pixel_y-dot_r),min(img.shape[0],pixel_y+dot_r+1)):
                pseudocolor_img[x,y,:] = color[:3]
    return pseudocolor_img


# From a sparse depth map, generate a pointcloud
def pointcloud_from_sparse_depth(sparse_depth_img,image_size,focal_length):
    sparse_depth_img = sparse_depth_img.astype(np.float32)
    sparse_depth_img = o3d.geometry.Image(sparse_depth_img)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=image_size, height=image_size, fx=image_size/2*focal_length, fy=image_size/2*focal_length, cx=image_size/2, cy=image_size/2)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(sparse_depth_img,intrinsic=intrinsic,depth_scale=1)
    return pcd

# From a sparse depth map, generate a pseudo-color pointcloud mesh (with each point as a small sphere)
def pseudo_colod_pointcloud_mesh(sparse_depth_img,sphere_r,depth_min,depth_max,image_size,focal_length,cmap=cm.terrain):
    norm = matplotlib.colors.Normalize(vmin=int(depth_min), vmax=int(depth_max), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    sparse_depth_img = sparse_depth_img.astype(np.float32)
    sparse_depth_img = o3d.geometry.Image(sparse_depth_img)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=image_size, height=image_size, fx=image_size/2*focal_length, fy=image_size/2*focal_length, cx=image_size/2, cy=image_size/2)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(sparse_depth_img,intrinsic=intrinsic,depth_scale=1)
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.zeros((pcd_points.shape[0],3))
    for i in range(pcd_colors.shape[0]):
        pcd_colors[i,:] = mapper.to_rgba(pcd_points[i,2])[:3]
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
    s.compute_vertex_normals()
    for i, p in enumerate(pcd.points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


# Attach a 2D texture image to a orthogonal mesh.
# inputs:
#   texture_map:    2D color map (H x W x 3)
#   mesh:           3D mesh in format open3d.geometry.TriangleMesh
#   focal_length:   The focal length of the camera to project the mesh.
#                   Useful to associate the mesh vertices to the texture map.
# outputs:
#   mesh:           3D textured mesh in format open3d.geometry.TriangleMesh
def texture_mesh(texture_map, mesh, focal_length):
    texture_map = o3d.geometry.Image(texture_map)
    mesh.textures = [texture_map]
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertices_2D = vertices[:,:2]/vertices[:,2:3]
    vertices_2D = (vertices_2D*focal_length+1)/2
    triangle_uvs = np.zeros((faces.shape[0]*3,2))
    for i in range(triangle_uvs.shape[0]):
        face_id = i//3
        face_v_id = i%3
        v_id = faces[face_id,face_v_id]
        triangle_uvs[i,:] = vertices_2D[v_id,:]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh

def texture_mesh_by_vertices(mesh, depth_min, depth_max, cmap=cm.terrain):
    norm = matplotlib.colors.Normalize(vmin=int(depth_min), vmax=int(depth_max), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
    mesh_v = np.asarray(mesh.vertices)
    vertex_colors = np.zeros((mesh_v.shape[0],3))
    for i in range(vertex_colors.shape[0]):
        vertex_colors[i,:] = mapper.to_rgba(mesh_v[i,2])[:3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)    
    return mesh

depth_missing_value = 74.5
depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
root_folder = "visualizations/journal/"
init_mesh_path = "visualizations/journal/cambridge_10_0186_init.obj"
refine_mesh_path = "visualizations/journal/cambridge_10_0186_refine.obj"
gt_mesh_path = "visualizations/journal/0186_gt_mesh.obj"
depth_img_path = "visualizations/journal/0186_gt_depth.png"
refine_depth_img_path = "visualizations/journal/cambridge_10_0186_refine_depth.png"
sparse_depth_img_path = "visualizations/journal/0186_sparse_depth.png"

sem_2d_img_path = "visualizations/journal/cambridge_10_0186_2D_sem.png"
sem_refine_img_path = "visualizations/journal/cambridge_10_0186_refine_sem.png"


save_pseudo_sparse_depth_img_path = "visualizations/journal/0186_pseudo_sparse_depth.png"
save_pseudo_depth_img_path = "visualizations/journal/0186_pseudo_gt_depth.png"
save_refine_depth_img_path = "visualizations/journal/cambridge_10_0186_pseudo_refine_depth.png"
save_sem_img_path = "visualizations/journal/gt_sem.png"

if __name__ == "__main__":
    

    mesh = o3d.io.read_triangle_mesh(refine_mesh_path)
    
    texture_map = imread(sem_2d_img_path)
    textured_mesh = texture_mesh(texture_map, mesh, focal_length)
    o3d.visualization.draw_geometries([textured_mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("visualizations/journal/temp/sem_mesh_0.obj",textured_mesh)

    texture_map = imread(sem_refine_img_path)
    textured_mesh = texture_mesh(texture_map, mesh, focal_length)
    o3d.visualization.draw_geometries([textured_mesh],mesh_show_wireframe=True,mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("visualizations/journal/temp/sem_mesh_1.obj",textured_mesh)
                