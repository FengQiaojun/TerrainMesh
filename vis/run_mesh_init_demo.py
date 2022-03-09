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

save_rgb_flat_mesh_path = "visualizations/journal/cambridge_10_0186_rgb_flat_pseudo.obj"
sparse_depth_img_path = "visualizations/journal/cambridge_10_0186_sparse_depth.png"

depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
#mean_depth = 70
sphere_r = 0.8
mesh_range_offset = 5
display_offset = 5

if __name__ == "__main__":

    # Generate sparse depth points
    sparse_depth_img = imread(sparse_depth_img_path)/depth_scale
    sparse_depth_mask = sparse_depth_img>0
    num_sparse_depth = np.sum(sparse_depth_mask)
    mean_depth = np.sum(sparse_depth_img)/num_sparse_depth
    pcd_sparse_depth = pointcloud_from_sparse_depth(sparse_depth_img,image_size,focal_length)
    pcd_mesh = pseudo_colod_pointcloud_mesh(sparse_depth_img,sphere_r,depth_min,depth_max,image_size,focal_length,cmap=cm.terrain)
    #o3d.io.write_triangle_mesh("visualizations/journal/mesh_init_sparse_depth.obj",pcd_mesh)
     
    # Generate flat mesh using the mean depth from sparse depth points
    vertices,faces = regular_512_1024()    
    vertices_2D = (vertices-image_size/2)/(image_size/2*focal_length)
    vertices = np.hstack((vertices_2D,np.ones((vertices.shape[0],1))))
    vertices *= mean_depth+mesh_range_offset
    #vertices[:,2] += 30
    textured_mesh_flat = o3d.geometry.TriangleMesh()
    textured_mesh_flat.vertices = o3d.utility.Vector3dVector(vertices)
    textured_mesh_flat.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(save_rgb_flat_mesh_path,textured_mesh_flat)
    textured_mesh_flat = o3d.io.read_triangle_mesh(save_rgb_flat_mesh_path)
    #textured_mesh_flat = texture_mesh_by_vertices(textured_mesh_flat,depth_min,depth_max)
    textured_mesh_flat.paint_uniform_color([0.7,0.7,0.7])
    # Offset for visualization
    textured_mesh_flat.translate([0,0,display_offset])

    # Project the pointcloud to the flat mesh
    # inputs:
    #   pcd_sparse_depth:   open3d.geometry.PointCloud
    #   mean_depth:         flat depth value
    pcd_points = np.asarray(pcd_sparse_depth.points)
    pcd_points_flat = np.copy(pcd_points)
    pcd_points_flat[:,2] = mean_depth + display_offset + mesh_range_offset
    pcd_points_pairs = np.vstack((pcd_points,pcd_points_flat))
    lines = np.vstack((np.arange(0,pcd_points_pairs.shape[0]/2),np.arange(pcd_points_pairs.shape[0]/2,pcd_points_pairs.shape[0]))).transpose()
    lines = lines.astype(int)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pcd_points_pairs),
        lines=o3d.utility.Vector2iVector(lines),
    )
    norm = matplotlib.colors.Normalize(vmin=int(depth_min), vmax=int(depth_max), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
    pcd_colors = np.zeros((pcd_points.shape[0],3))
    for i in range(pcd_colors.shape[0]):
        pcd_colors[i,:] = mapper.to_rgba(pcd_points[i,2])[:3]
    line_set.colors = o3d.utility.Vector3dVector(pcd_colors)

    line_mesh1 = LineMesh(pcd_points_pairs, lines, pcd_colors, radius=0.1)
    line_mesh1_geoms = line_mesh1.cylinder_segments

    textured_mesh_flat_frame = o3d.geometry.LineSet.create_from_triangle_mesh(textured_mesh_flat)
    o3d.visualization.draw_geometries([pcd_mesh,textured_mesh_flat_frame,*line_mesh1_geoms],mesh_show_back_face=True)
    
    #final_mesh = pcd_mesh + textured_mesh_flat
    #for line in line_mesh1_geoms:
    #    final_mesh += line
    #o3d.io.write_triangle_mesh("visualizations/journal/mesh_init_visualization.obj",final_mesh)