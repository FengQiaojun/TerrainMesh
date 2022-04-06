
### TODO

### Existing models:

Geo:  
0309_1635_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_channel5_2_3_32_0.0005  
Geo w/o init:  
0310_0958_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_channel5_2_3_32_0.0005  
Geo w/o norm:  
0310_0959_resnet18_train_mesh1024_depth[1000]_2D_3D_channel5_2_3_32_0.0005  
Geo RGB+RD:  
0311_1848_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_channel4_2_3_32_0.0005  
Geo RGB:  
0311_1849_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_channel3_2_3_32_0.0005  
Geo RD+EDT:  
0311_2212_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_channel2_2_3_32_0.0005
Geo Mesh 576:  
0312_1051_resnet18_train_mesh576_depth[1000]_dnorm_2D_3D_channel5_2_3_32_0.0005  
Geo Mesh 2025:  
0312_2121_resnet18_train_mesh2025_depth[1000]_dnorm_2D_3D_channel5_2_3_32_0.0005  

Entropy Hybrid:  
0311_1817_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_Semantic_CrossEntropy_channel5_2_3_32_0.0005
Focal Hybrid:  
0311_0756_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_Semantic_Focal_channel5_2_3_32_0.0005  
Jaccard Hybrid:  
0310_1531_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_Semantic_Jaccard_channel5_2_3_32_0.0005  
Dice Hybrid:  
0311_0758_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_Semantic_Dice_channel5_2_3_32_0.0005  
Dice no residual:  
0315_1610_resnet18_train_mesh1024_depth[1000]_dnorm_2D_3D_Semantic_Dice_channel5_2_3_32_0.0005




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


conda list --explicit > terrainmesh.txt
### Dependencies
pytorch  
pytorch3d  
open3d  
fvcore  
imageio  
matplotlib


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Zj4DgLTgv2tEzjmioF381jSMUHebogrq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Zj4DgLTgv2tEzjmioF381jSMUHebogrq" -O checkpoints.zip && rm -rf /tmp/cookies.txt