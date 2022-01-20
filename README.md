TODO: Refine the Meshes groundtruth for a few specific large error ones.

terrainmesh.txt:  
This is the export file of a conda environment. 



**dataset**  
To build the dataloader.

**mesh_init**  
Include functions that initialize a 3D mesh from a flat mesh using sparse or dense depth map. One use linear solver when the depth measurements are limited. The other use PyTorch3D differentiable renderer to back-propagate the error on 2D depth image.  

**model**  
TODO: try the Deeplab version image backbone and classification module.  
The neural network model. 