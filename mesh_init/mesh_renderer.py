# render the mesh using a customized naive shader that preserve the vertex color without any other effects.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

class TextureShader(nn.Module):
    """
    A super simple shader that directly attach texture to the image 
    """
    def __init__(
        self, device="cpu"):
        super().__init__()
        
    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        colors = texels
        # remove the dimension of K which is the number of triangles/surfaces
        N, H, W, K, C = colors.shape
        if K > 1:
            colors = colors[:,:,:,0,:]
        images = torch.squeeze(colors,dim=3)
        return images

def render_mesh_vertex_texture(verts,faces,feats,image_size=512,device=None):
    textures = TexturesVertex(verts_features=feats.to(device))
    #mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    mesh.textures = textures
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=-1,)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.00001,
        faces_per_pixel=1, 
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    shader = TextureShader(device=device)
    
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer, shader=shader
    )
    images, depth = renderer(mesh)

    return images, depth

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

def mesh_render_depth(mesh,image_size=512,focal_length=-10,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    # Define the camera
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length)
    # Define the Rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0001, 
        faces_per_pixel=1, 
    )
    # Define the renderer
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
    )
    depth = renderer(mesh)
    depth = depth[0, ..., 0].detach().cpu().numpy()
    # fill the empty region with mean
    depth_mean = np.mean(depth[np.where(depth>0)])
    depth[np.where(depth<=0)] = depth_mean
    return depth

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    verts = torch.tensor(
                [
                    [ 0.5, -0.5,  0.0],
                    [ 0.5,  0.5,  0.0],
                    [-0.5, -0.5,  0.0],
                    [-0.5,  0.5,  0.0],
                    [ 0.0,  0.0, -0.2],
                ],
                dtype=torch.float32,device=device
    )
    faces = torch.tensor(
                [
                    [0, 4, 1],
                    [1, 4, 3],
                    [3, 4, 2],
                    [2, 4, 0],
                ],
                dtype=torch.int64,device=device
    )
    rgb = torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.5],
                    [1.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0.5],
                    [1.0, 1.0, 0.0, 0.5],
                    [1.0, 1.0, 0.0, 0.5],
                ],
                dtype=torch.float32,device=device
    )

    rgb = rgb[None]
    textures = TexturesVertex(verts_features=rgb.to(device))

    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)

    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    T[0,2] = 0.7
    cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=-1,)
    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=0.00001,
        faces_per_pixel=1, 
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    shader = TextureShader(device=device)
    
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer, shader=shader
    )
    images, depth = renderer(mesh)

    depth = depth.zbuf
    print(images.shape)
    images_display = images[0, ..., :3].cpu().numpy()
    depth_display = depth[0, ..., 0].cpu().numpy()

    plt.imshow(images_display)
    plt.grid("off")
    plt.axis("off")
    plt.show()
    