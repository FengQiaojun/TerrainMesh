### Functions

**meshing.py**  
Generate a regular grid-like 2D mesh with equal sized triangles. Now we have 576 vertices and 1024 vertices versions.  

**mesh_init_linear_solver.py**  
Initialize a mesh with sparse depth measurments. Formulate as a linear solver.

**mesh_opt.py**  
Some functions used to optimize mesh using 2D depth map. 


### Scripts

**run_calc_edt.py**
Calculate Euclidean Distance Transform for the sparse depth map.  

**run_mesh_init_sparse_depth.py**  
Use this to generate some initial meshes. Also re-sample the sparse depth to get a fixed-number of points there.

**run_mesh_gt_depth.py**  
Initialize a super dense mesh as the groundtruth. We actually sample points to generate pointcloud as the groundtruth.