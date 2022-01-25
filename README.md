TODO: 
* A simpler model for segmentation. Using resnet 34 or resnet 18.


* Get initial number of chamfer error and depth error.
* Comparison between
    a. First train from scratch on semantic segmentation, then geometry
    b. First train with pretrained weight, then geometry
    c. Train all from scratch.


terrainmesh.txt:  
This is the export file of a conda environment. 


### Existing models:
0124_2331_train_mesh1024_depth1000_channel3_focal_loss_50_0.01  
The model for semantic segmentation initialization. Takes 3 channels as inputs.  

0124_1722_train_mesh1024_depth1000_channel4_focal_loss_50_0.01
The model for semantic segmentation initialization. Takes 4 channels as inputs.    


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