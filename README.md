
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
If you use our dataset