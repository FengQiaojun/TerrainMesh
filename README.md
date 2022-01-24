TODO: 
1. Write a pure semantic segmentation training using Deeplab V3. Use that to initialize the feature extration network.
2. Get initial number of chamfer error and depth error.
3. Comparison between
    a. First train from scratch on semantic segmentation, then geometry
    b. First train with pretrained weight, then geometry
    c. Train all from scratch.


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

**segmentation**  
Ref: https://github.com/VainF/DeepLabV3Plus-Pytorch  


**utils**  
Some utility functions.