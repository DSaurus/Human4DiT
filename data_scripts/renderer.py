from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
    BlendParams,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)
import torch
import torch.nn as nn
import numpy as np

class PlainColorShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blend_params = BlendParams(background_color=(0., 0., 0.), sigma=1e-20, gamma=1e-20)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        #vert_colors = meshes.verts_normals_packed() * 0.5 + 0.5
        vert_colors = kwargs.get("vert_colors")
        faces = meshes.faces_packed()
        face_colors = vert_colors[faces]
        colors = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_colors
        )

        blend_params = kwargs.get("blend_params", self.blend_params)
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images


def render_smpl_vcano_map(smpl_verts, smpl_faces, device, cam_R=torch.eye(3), cam_t=torch.zeros((3,1)), cam_K=torch.zeros((3,3)), img_w=1024, img_h=1024):
    cam_f = torch.stack([cam_K[0, 0], cam_K[1, 1]], dim=-1)
    cam_c = torch.stack([img_w - cam_K[0, 2], img_h - cam_K[1, 2]], dim=-1)
    smpl_verts = torch.from_numpy(smpl_verts.astype(np.float32)).unsqueeze(0).to(device)
    smpl_faces = torch.from_numpy(smpl_faces.astype(np.float32)).to(device)

    cameras = PerspectiveCameras(
        focal_length=cam_f.reshape(-1, 2), principal_point=cam_c.reshape(-1, 2),
        R=cam_R.reshape(-1, 3, 3).permute(0, 2, 1), T=cam_t.reshape(-1, 3),
        # R=cam_R.reshape(-1, 3, 3), T=cam_t.reshape(-1, 3),
        in_ndc=False, image_size=((img_h, img_w),),
        device=smpl_verts.device)


    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=len(smpl_faces),
    )
    
    render_meshes = Meshes(verts=smpl_verts, faces=smpl_faces.unsqueeze(0))

    
    vert_colors = render_meshes.verts_normals_packed() * -1
    vert_colors = cam_R.reshape(-1, 3, 3).to(device) @ vert_colors.transpose(0, 1).unsqueeze(0)
    vert_colors = vert_colors[0].transpose(0, 1)
    vert_colors = vert_colors* 0.5 + 0.5

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=PlainColorShader()
    )

    cano_vert_render = renderer(render_meshes, vert_colors=vert_colors)
    cano_vert_render = torch.flip(cano_vert_render, [1, 2])
    return cano_vert_render