from imageio import imread
import matplotlib.cm as cm
import numpy as np
import open3d as o3d

from vis import pseudo_colod_pointcloud_mesh

sphere_r = 0.5
depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
root_folder = "visualizations/journal/"

if __name__ == "__main__":
    
    # Test pseudo_colod_pointcloud_mesh
    sparse_depth_img = imread(root_folder+"cambridge_10_0186_sparse_depth.png")/depth_scale
    
    sparse_depth_mask = sparse_depth_img>0
    num_sparse_depth = np.sum(sparse_depth_mask)
    mean_depth = np.sum(sparse_depth_img)/num_sparse_depth
    print(mean_depth)

    pcd_mesh = pseudo_colod_pointcloud_mesh(sparse_depth_img,sphere_r,depth_min,depth_max,image_size,focal_length,cmap=cm.terrain)
    o3d.visualization.draw_geometries([pcd_mesh])
    o3d.io.write_triangle_mesh(root_folder+"cambridge_10_0186_sparse_depth.obj",pcd_mesh)