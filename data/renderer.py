from data.label_conversions import convert_multiclass_to_binary_labels_torch

import pytorch3d
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)


def renderer(vertices, faces, R, T, device):
    mesh = Meshes(verts=vertices, faces=faces[None])

    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    # R, T = look_at_view_transform(2.7, 0, 90) 
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Rasterization settings for silhouette rendering  
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Silhouette renderer 
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    # Render silhouette images.  The 3rd channel of the rendering output is 
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(mesh, cameras=cameras, lights=lights)
    silhouette_image = silhouette_images[:, :, :, 3]
    
    silhouette_image = convert_multiclass_to_binary_labels_torch(silhouette_image)

    return silhouette_image