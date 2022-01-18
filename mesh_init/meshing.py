# include some mesh building functions
import numpy as np 
from scipy.spatial import Delaunay
import torch
from pytorch3d.structures import Meshes

def regular_512_576():
    x = np.linspace(-2,513,24)
    y = np.linspace(-2,513,24)
    xx, yy = np.meshgrid(x, y)
    vertices = np.concatenate((xx[..., np.newaxis],yy[..., np.newaxis]),axis=-1)
    vertices = np.reshape(vertices,(-1,2))
    tri = Delaunay(vertices)
    faces = tri.simplices
    mesh = Meshes(verts=[torch.tensor(vertices)], faces=[torch.tensor(faces)])
    Laplacian = mesh.laplacian_packed().to_dense().numpy()
    return vertices,faces,Laplacian
    
def regular_512_1024():
    #x = np.linspace(8,504,32)
    #y = np.linspace(8,504,32)
    x = np.linspace(-2,513,32)
    y = np.linspace(-2,513,32)
    xx, yy = np.meshgrid(x, y)
    vertices = np.concatenate((xx[..., np.newaxis],yy[..., np.newaxis]),axis=-1)
    vertices = np.reshape(vertices,(-1,2))
    tri = Delaunay(vertices)
    faces = tri.simplices
    mesh = Meshes(verts=[torch.tensor(vertices)], faces=[torch.tensor(faces)])
    Laplacian = mesh.laplacian_packed().to_dense().numpy()
    return vertices,faces,Laplacian