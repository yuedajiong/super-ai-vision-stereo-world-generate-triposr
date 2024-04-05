import torch
import einops
import numpy as np
import math
import PIL

from network_tokenizer import DINOSingleImageTokenizer, Triplane1DTokenizer
from network_backbone import Transformer1D, TriplaneUpsampleNetwork
from network_nerf_decoder import NeRFMLP
from network_nerf_renderer import TriplaneNeRFRenderer

class TSR(torch.nn.Module):
    def __init__(self, radius, valid_thresh, num_samples_per_ray, img_size, depth, embed_dim, num_channels, num_layers, cross_attention_dim, n_hidden_layers, official):
        super().__init__()
        self.image_tokenizer = DINOSingleImageTokenizer(img_size, depth, embed_dim, official=official)
        self.tokenizer = Triplane1DTokenizer(num_channels=num_channels)
        self.backbone = Transformer1D(num_channels=num_channels, num_layers=num_layers, cross_attention_dim=cross_attention_dim)
        self.post_processor = TriplaneUpsampleNetwork(in_channels=num_channels, out_channels=40)
        self.decoder = NeRFMLP(n_hidden_layers=n_hidden_layers)
        self.renderer = TriplaneNeRFRenderer(radius=radius, valid_thresh=valid_thresh, num_samples_per_ray=num_samples_per_ray)

    def forward(self, rgb_cond):
        input_image_tokens = self.image_tokenizer(einops.rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1))  #[-1, 1, 768, 1025]
        input_triplane_tokens = self.tokenizer(rgb_cond.shape[0])  #[-1, 1024, 3072] 
        mixed_tokens = self.backbone(input_triplane_tokens, encoder_hidden_states=einops.rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1))  #[-1, 1024, 3072]
        scene_codes = self.post_processor(self.tokenizer.detokenize(mixed_tokens))  #[-1, 3, 40, 64, 64]
        return scene_codes

    def extract_mesh(self, scene_codes, resolution=256, threshold=25.0):
        import torchmcubes  #wget https://github.com/tatsy/torchmcubes/archive/refs/heads/master.zip ; pip install ./torchmcubes
        class MarchingCubeIsosurfaceHelper(torch.nn.Module):
            def __init__(self, resolution, points_range=(0, 1)):
                super().__init__()
                self.resolution = resolution
                self.points_range = points_range
                x, y, z = torch.meshgrid((torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)), indexing="ij")
                self.grid_vertices = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)  #keep the vertices on CPU so that we can support very large resolution

            def forward(self, level):
                v_pos, t_pos_idx = torchmcubes.marching_cubes(-level.view(self.resolution, self.resolution, self.resolution).detach(), 0.0)
                v_pos = v_pos[..., [2, 1, 0]] / (self.resolution - 1.0)
                return v_pos, t_pos_idx

        def scale_tensor(dat, inp_scale, tgt_scale):
            if inp_scale is None:
                inp_scale = (0, 1)
            if tgt_scale is None:
                tgt_scale = (0, 1)
            if isinstance(tgt_scale, torch.FloatTensor):
                assert dat.shape[-1] == tgt_scale.shape[-1]
            dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
            dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
            return dat

        isosurface_helper = MarchingCubeIsosurfaceHelper(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(self.decoder, scale_tensor(isosurface_helper.grid_vertices.to(scene_codes.device), isosurface_helper.points_range, (-self.renderer.radius, self.renderer.radius)), scene_code)["density_act"]
            v_pos, t_pos_idx = isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(v_pos, isosurface_helper.points_range, (-self.renderer.radius, self.renderer.radius))
            with torch.no_grad():
                color = self.renderer.query_triplane(self.decoder, v_pos, scene_code)["color"]
            import trimesh  #pip install trimesh
            mesh = trimesh.Trimesh(vertices=v_pos.cpu().numpy(), faces=t_pos_idx.cpu().numpy(), vertex_colors=color.cpu().numpy())
            meshes.append(mesh)
        return meshes

    def render_images(self, scene_codes, n_views, elevation_deg=0.0, camera_distance=1.9, fovy_deg=40.0, height=256, width=256, return_type="pt"):
        def get_spherical_cameras(n_views: int, elevation_deg: float, camera_distance: float, fovy_deg: float, height: int, width: int):
            def get_ray_directions(H: int, W: int, focal, principal=None, use_pixel_centers=True, normalize=True):
                pixel_center = 0.5 if use_pixel_centers else 0

                if isinstance(focal, float):
                    fx, fy = focal, focal
                    cx, cy = W / 2, H / 2
                else:
                    fx, fy = focal
                    assert principal is not None
                    cx, cy = principal

                i, j = torch.meshgrid(
                    torch.arange(W, dtype=torch.float32) + pixel_center,
                    torch.arange(H, dtype=torch.float32) + pixel_center,
                    indexing="xy",
                )

                directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)

                if normalize:
                    directions = torch.nn.functional.normalize(directions, dim=-1)

                return directions

            def get_rays(directions, c2w, keepdim=False, normalize=False):
                # Rotate ray directions from camera coordinate to the world coordinate
                assert directions.shape[-1] == 3

                if directions.ndim == 2:  # (N_rays, 3)
                    if c2w.ndim == 2:  # (4, 4)
                        c2w = c2w[None, :, :]
                    assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
                    rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
                    rays_o = c2w[:, :3, 3].expand(rays_d.shape)
                elif directions.ndim == 3:  # (H, W, 3)
                    assert c2w.ndim in [2, 3]
                    if c2w.ndim == 2:  # (4, 4)
                        rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                            -1
                        )  # (H, W, 3)
                        rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
                    elif c2w.ndim == 3:  # (B, 4, 4)
                        rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                            -1
                        )  # (B, H, W, 3)
                        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
                elif directions.ndim == 4:  # (B, H, W, 3)
                    assert c2w.ndim == 3  # (B, 4, 4)
                    rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                        -1
                    )  # (B, H, W, 3)
                    rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

                if normalize:
                    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
                if not keepdim:
                    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

                return rays_o, rays_d

            azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
            elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
            camera_distances = torch.full_like(elevation_deg, camera_distance)

            elevation = elevation_deg * math.pi / 180
            azimuth = azimuth_deg * math.pi / 180

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

            # default scene center at origin
            center = torch.zeros_like(camera_positions)
            # default camera up direction as +z
            up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)

            fovy = torch.full_like(elevation_deg, fovy_deg) * math.pi / 180

            lookat = torch.nn.functional.normalize(center - camera_positions, dim=-1)
            right = torch.nn.functional.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
            up = torch.nn.functional.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
            c2w3x4 = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
            c2w[:, 3, 3] = 1.0

            # get directions by dividing directions_unit_focal by focal length
            focal_length = 0.5 * height / torch.tan(0.5 * fovy)
            #print('$$$$$$$$$ focal_length', focal_length)
            directions_unit_focal = get_ray_directions(
                H=height,
                W=width,
                focal=1.0,
            )
            directions = directions_unit_focal[None, :, :, :].repeat(n_views, 1, 1, 1)
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )
            # must use normalize=True to normalize directions here
            rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

            return rays_o, rays_d

        rays_o, rays_d = get_spherical_cameras(n_views, elevation_deg, camera_distance, fovy_deg, height, width)
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return PIL.Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
            else:
                raise NotImplementedError
        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image, alpha = self.renderer(self.decoder, scene_code, rays_o[i], rays_d[i])
                images_.append(process_output(image))
            images.append(images_)
        return images
