from imageio import imread
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from transforms3d.quaternions import quat2mat
import os

from vis import pseudo_color_map, texture_mesh, texture_mesh_by_vertices
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


depth_min = 50
depth_max = 80
focal_length = 2
image_size = 512
depth_scale = 100
mean_depth = 70

seq_name = "cambridge_10"
kf_list = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"+seq_name+"/kf_"+seq_name+"_scaled.txt"
input_folder = "/mnt/NVMe-2TB/qiaojun/journal_terrain/visualizations/journal/"+seq_name
sample_per_frame = 4
boundary_removal = 0.00
cam_frame_list = []
intrinsic = np.array([[512,0,256],[0,512,256],[0,0,1]])
frame_scale = 10
frame_radius = 0.3
old_trans = None

if __name__ == "__main__":

    mesh_list = []

    f = open(kf_list,"r")
    frame_count = 0
    for line in f.readlines():
        frame_count+=1
        if frame_count%sample_per_frame != 0:
            continue
        line = line.split()
        img_idx = int(line[0])
        translation = np.array(line[1:4])
        rotation = quat2mat([float(line[7]), float(line[4]), float(line[5]), float(line[6])])
        trans = np.eye(4)
        trans[0:3,0:3] = rotation
        trans[0:3,3] = translation
        print(img_idx)
        #print(img_idx,trans)

        cam_frame = o3d.geometry.LineSet.create_camera_visualization(512, 512, intrinsic, np.linalg.inv(trans), scale=frame_scale)
        cam_frame = LineMesh(cam_frame, radius=frame_radius)
        cam_frame_geoms = cam_frame.cylinder_segments
        cam_frame_list.extend(cam_frame_geoms)
        if type(old_trans) is np.ndarray:
            p1 = o3d.geometry.PointCloud()
            p1.points = o3d.utility.Vector3dVector(np.expand_dims(old_trans[:3,3],0))
            p2 = o3d.geometry.PointCloud()
            p2.points = o3d.utility.Vector3dVector(np.expand_dims(trans[:3,3],0))
            traj_frame = o3d.geometry.LineSet.create_from_point_cloud_correspondences(p1,p2,[(0,0)])
            traj_frame.paint_uniform_color([1,0,0])
            traj_mesh = LineMesh(traj_frame, radius=frame_radius)
            traj_mesh_geoms = traj_mesh.cylinder_segments
            cam_frame_list.extend(traj_mesh_geoms)
        old_trans = trans


        mesh = o3d.io.read_triangle_mesh(os.path.join(input_folder,seq_name+"_%04d_refine.obj"%img_idx))
        texture_map = imread(os.path.join(input_folder,seq_name+"_%04d_refine_sem.png"%img_idx))
        #texture_map = imread(os.path.join(input_folder,seq_name+"_%04d_init_sem.png"%img_idx))
        #texture_map = imread("/mnt/NVMe-2TB/qiaojun/SensatUrban/"+seq_name+"/Images/%04d.png"%img_idx)
        
        # Crop the boundary
        bbox_origin = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh.vertices)
        bbox_origin_min = bbox_origin.get_min_bound()
        bbox_origin_max = bbox_origin.get_max_bound()
        bbox_origin_gap = bbox_origin_max - bbox_origin_min
        bbox_shrink = o3d.geometry.AxisAlignedBoundingBox(min_bound=[bbox_origin_min[0]+boundary_removal*bbox_origin_gap[0],bbox_origin_min[1]+boundary_removal*bbox_origin_gap[1],bbox_origin_min[2]],max_bound=[bbox_origin_max[0]-boundary_removal*bbox_origin_gap[0],bbox_origin_max[1]-boundary_removal*bbox_origin_gap[1],bbox_origin_max[2]])
        mesh = mesh.crop(bbox_shrink)
        
        textured_mesh = texture_mesh(texture_map, mesh, focal_length)
        textured_mesh.transform(trans)
        o3d.io.write_triangle_mesh("temp.obj",textured_mesh)
        textured_mesh = o3d.io.read_triangle_mesh("temp.obj", True)
        textured_mesh.compute_vertex_normals()
        
        
        mesh_list.append(textured_mesh)
        #o3d.visualization.draw_geometries(mesh_list,mesh_show_wireframe=False,mesh_show_back_face=True)
        
        #if img_idx > 50:
        #    break
    print(len(mesh_list))
    o3d.visualization.draw_geometries(mesh_list+cam_frame_list,mesh_show_wireframe=False,mesh_show_back_face=True)

