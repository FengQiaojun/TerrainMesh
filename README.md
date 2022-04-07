
# TerrainMesh: Metric-Semantic Terrain Reconstruction from Aerial Images Using Joint 2D-3D Learning

Check this demo on Colab!  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sRMjztDcZaKHfvkV3A2YurW9F4sojoel?usp=sharing)

### Install the dependencies
We use Conda to manage the packages. The exact package we use are provided. You can create a conda environment by
```
$ conda create --name <env> --file terrainmesh.txt
$ conda activate <env>
```

### Run the demo
A simple demo is provided in [demo.ipynb](demo.ipynb), which might need a local GPU.  
Another option is to try the [Google Colab demo](https://colab.research.google.com/drive/1sRMjztDcZaKHfvkV3A2YurW9F4sojoel?usp=sharing), which runs on Google's GPU.



### Reference 
```bibtex
@misc{Feng2022TerrainMesh,
  author = {Feng, Qiaojun and Atanasov, Nikolay},
  title = {TerrainMesh: Metric-Semantic Terrain Reconstruction from Aerial Images Using Joint 2D-3D Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
```bibtex
@INPROCEEDINGS{Feng2021Mesh,
  author={Feng, Qiaojun and Atanasov, Nikolay},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Mesh Reconstruction from Aerial Images for Outdoor Terrain Mapping Using Joint 2D-3D Learning}, 
  year={2021},
  pages={5208-5214},
  doi={10.1109/ICRA48506.2021.9561337}}
```
If you use the [datasets](https://github.com/FengQiaojun/TerrainMesh_Data) in the paper, consider citing  
[WHU MVS/Stereo Dataset](http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html)
```bibtex
@INPROCEEDINGS{Liu2020WHU,
  author={Liu, Jin and Ji, Shunping},
  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={A Novel Recurrent Encoder-Decoder Structure for Large-Scale Multi-View Stereo Reconstruction From an Open Aerial Dataset}, 
  year={2020},
  pages={6049-6058},
  doi={10.1109/CVPR42600.2020.00609}}
```
[SensatUrban Dataset](http://point-cloud-analysis.cs.ox.ac.uk/)
```bibtex
@article{Hu2021Sensat,
	doi = {10.1007/s11263-021-01554-9},
	year = 2022,
	month = {Jan},
	volume = {130},
	number = {2},
	pages = {316--343},
	author = {Qingyong Hu and Bo Yang and Sheikh Khalid and Wen Xiao and Niki Trigoni and Andrew Markham},
	title = {{{SensatUrban}: Learning Semantics from Urban-Scale Photogrammetric Point Clouds}},
	journal = {International Journal of Computer Vision}
}
```