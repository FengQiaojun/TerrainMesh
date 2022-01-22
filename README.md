TODO: Refine the Meshes groundtruth for a few specific large error ones.

terrainmesh.txt:  
This is the export file of a conda environment. 


**train.py**  
The main training function.  

**loss.py**  
Define the loss functions we use. Include the 2D/3D for geometric reconstruction and cross entropy for semantic segmentation.  

**config.py**  
Some configuration parameters. All important adjustable parameters should also be included in Sensat_basic.yaml.  

**dataset**  
To build the dataloader.

**mesh_init**  
Include functions that initialize a 3D mesh from a flat mesh using sparse or dense depth map. One use linear solver when the depth measurements are limited. The other use PyTorch3D differentiable renderer to back-propagate the error on 2D depth image.  

**model**  
TODO: try the Deeplab version image backbone and classification module.  
The neural network model. 

**utils**  
Some utility functions.