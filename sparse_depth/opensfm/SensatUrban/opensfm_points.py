# Run OpenSfM and build sparse point cloud from 5 images.
# Inputs: 5 RGB images and initial camera intrinsic
# Outputs: the sparse point cloud in the middle reference image frame.
# Step 1: run the OpenSfM
# Step 2: rescale the map by calculating the ratio between the real translation and the recovered translation (TODO: how to set the known camera intrinsic + extrinsic)
# Step 3: refine the point cloud alignment using the groundtruth RGBD image
# Step 4: generate the point cloud, sparse depth map(rounded-pixels), and potentially the quantitative error
# Iterate this for all pairs

import numpy as np 
import os 
import sys
from imageio import imread,imwrite
import matplotlib.pyplot as plt
import json
from transforms3d.euler import euler2mat
import open3d as o3d 
import cv2
import copy

dataset_index = "birmingham_2"
input_folder = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"+dataset_index+"/Images"
input_depth_folder = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"+dataset_index+"/Depths"
output_folder = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"+dataset_index+"/Pcds"
pair_list = "pairs.txt"
command_list = ["extract_metadata","detect_features","match_features","create_tracks","reconstruct","export_ply"]
depth_error_threshold = 10

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
f = open(pair_list,"r")
pair_info = f.readlines()

for pair in pair_info[0:1]:
    # read the 5 images names
    pair_img = pair.split()
    print(pair_img)
    available_cam = 0
    for p in pair_img:
        if p != "xxxx.png":
            available_cam += 1
    # create a temp folder to store outputs of OpenSfM
    #os.mkdir("temp/"+dataset_index)
    os.makedirs("temp"+dataset_index+"/images")
    # copy the 5 images
    for img in pair_img:
        if img != "xxxx.png":
            img_path_from = os.path.join(input_folder,img)
            img_path_to = os.path.join("temp"+dataset_index+"/images",img)
            os.system(f"cp {img_path_from} {img_path_to}")
    os.system(f"cp camera_models_overrides.json temp"+dataset_index+"/camera_models_overrides.json")
    depth_error = 100
    # Try multiple time of OpenSfM when the error is large
    while depth_error >= depth_error_threshold:
        # run the sequence of OpenSfM commands
        for command in command_list:
            os.system(f"python3 /mnt/NVMe-2TB/qiaojun/journal_terrain/OpenSfM/bin/opensfm {command} temp"+dataset_index)
        # rescale the map
        json_file = "temp"+dataset_index+"/reconstruction.json"
        with open(json_file) as f:
            data = json.load(f)[0]
        cams = data["shots"]
        if len(cams) < available_cam:
            continue
        ratio_list = []
        cam_T = np.zeros((5,4,4))
        for i in range(5):
            cam_T[i,...] = np.eye(4)
            if pair_img[i] != "xxxx.png":
                euler = cams[pair_img[i]]["rotation"]
                cam_T[i,:3,:3] = euler2mat(euler[0],euler[1],euler[2])
                cam_T[i,:3,3] = cams[pair_img[i]]["translation"]
                if i > 0:
                    T_ref = cam_T[i,...]@np.linalg.inv(cam_T[0,...])
                    if i in [1,2]:
                        trans_ref = T_ref[0,3]
                        ratio = np.abs(7/trans_ref)
                    #elif i in [3,4]:
                    #    trans_ref = T_ref[1,3]
                    #    ratio = np.abs(25/trans_ref)
                    ratio_list.append(ratio)
        ratio = np.mean(ratio_list)
        base_R = cam_T[0,:3,:3]
        base_t = cam_T[0,:3,3]
        feature_points = data["points"]
        n_points = len(feature_points)
        points = np.zeros((n_points,3))
        colors = np.zeros((n_points,3))
        count_idx = 0
        for _,p in feature_points.items():
            points[count_idx,:] = p["coordinates"]
            points[count_idx,:] = base_R@(points[count_idx,:])+base_t
            colors[count_idx,:] = p["color"]
            colors[count_idx,:] /= 255
            count_idx += 1
        points *= ratio
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        #o3d.visualization.draw_geometries([pcd])
        # convert to 2D sparse depth map
        image_size = (512,512)
        image_c = 256
        image_f = 512
        depth_image = np.zeros(image_size)
        for i in range(points.shape[0]):
            pt = points[i,:2]/points[i,2]
            pt = np.round(pt*image_f+image_c).astype(np.int)
            if pt[0]<0 or pt[0]>511 or pt[1]<0 or pt[1]>511:
                #print(pt)
                continue
            depth_image[pt[1],pt[0]] = points[i,2]
        depth_image_ref = imread(os.path.join(input_depth_folder,pair_img[0]))/64
        depth_visible = depth_image>0
        #plt.imshow(depth_visible)
        #plt.show()
        if np.sum(depth_visible) == 0:
            continue
        depth_error = np.sum(np.abs(depth_image-depth_image_ref)*depth_visible) / np.sum(depth_visible)
        print(depth_error)
        
    # Use the colored point cloud registration to optimize the registration result.
    radius = 2
    max_iter = 50
    source = pcd 
    rgb_image_ref = imread(os.path.join(input_folder,pair_img[0]))
    depth_image_ref = imread(os.path.join(input_depth_folder,pair_img[0]))/64
    target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=o3d.geometry.Image(rgb_image_ref.astype(np.uint8)), depth=o3d.geometry.Image(depth_image_ref.astype(np.uint16)), depth_scale=1, depth_trunc=700, convert_rgb_to_intensity=False)
    cam_int = o3d.camera.PinholeCameraIntrinsic(512,512,2560,2560,256,256)
    target = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd,cam_int)
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    current_transformation = np.identity(4)
    result_icp = o3d.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                relative_rmse=1e-6,
                                                max_iteration=max_iter))
    current_transformation = result_icp.transformation
    source.transform(current_transformation)
    #o3d.visualization.draw_geometries([source, target])
    # Calculate the depth error and the rgb error
    points = np.array(source.points)
    depth_image = np.zeros(image_size)
    #rgb_image = np.zeros(image_size+(3,))
    rgb_image = imread(os.path.join(input_folder,pair_img[0]))
    for i in range(points.shape[0]):
        pt = points[i,:2]/points[i,2]
        pt = np.round(pt*image_f+image_c).astype(np.int)
        if pt[0]<0 or pt[0]>511 or pt[1]<0 or pt[1]>511:
            #print(pt)
            continue
        depth_image[pt[1],pt[0]] = points[i,2]
        cv2.circle(rgb_image,(pt[0],pt[1]),radius=3,color=colors[i,:]*255,thickness=-1)
    depth_image_ref = imread(os.path.join(input_depth_folder,pair_img[0]))/64
    depth_visible = depth_image>0
    if np.sum(depth_visible) == 0:
        continue
    depth_error = np.sum(np.abs(depth_image-depth_image_ref)*depth_visible) / np.sum(depth_visible)
    print("Final depth error", depth_error)

    # save color point cloud as well as the sparse depth map
    imwrite(os.path.join(output_folder,pair_img[0][:-4]+"_rgb.png"),rgb_image.astype(np.uint8))
    imwrite(os.path.join(output_folder,pair_img[0]),(depth_image*64).astype(np.uint16))
    o3d.io.write_point_cloud(os.path.join(output_folder,pair_img[0][:-4]+".ply"),source)
    # remove the temp folder
    os.system("rm -r temp"+dataset_index)
