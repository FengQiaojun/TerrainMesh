Training Lessons:
* Seems like the package versions matter (PyTorch, PyTorch3D)
  * PyTorch 1.6.0 + PyTorch3D 0.2.5 (Promising.)
  * PyTorch 1.6.0 + PyTorch3D 0.3.0 (Promising.)
  * PyTorch 1.6.0 + PyTorch3D 0.4.0 (Tried very few epochs but seems not working well.)
  * PyTorch 1.7.1 + PyTorch3D 0.4.0 (Tried most. Not able to train stably.)
  * PyTorch 1.7.1 + PyTorch3D 0.6.1 (Tried very few epochs but seems not working well.)

* For Depth Only, Normalized Mesh, 1000 samples  
  * Learning rate should be 1e-6. 5e-6 will be unstable.




TODO: 
* Find a way to introduce semantic without sacrificing the geometric performance.
  Maybe use a separated GNN solely for geometric.
* Also initialize the feature extractor and classifier modules. Whether we should train them?

* Tune loss weight parameters. Start with only one or a few test cases to see what's going on.



terrainmesh.txt:  
This is the export file of a conda environment. Generated by conda list --explicit > terrainmesh.txt  
Build an env from this: conda create --name terrainmesh --file terrainmesh.txt

### Existing models:
0124_2331_train_mesh1024_depth1000_channel3_focal_loss_50_0.01  
The resnet50 model for semantic segmentation initialization. Takes 3 channels as inputs.  

0125_1617_deeplab_resnet18_train_mesh1024_depth1000_channel3_focal_loss_50_0.01  
The resnet18 model for semantic segmentation initialization. Takes 3 channels as inputs.  

0125_1619_deeplab_resnet34_train_mesh1024_depth1000_channel3_focal_loss_50_0.01
The resnet34 model for semantic segmentation initialization. Takes 3 channels as inputs.  

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