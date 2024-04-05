import torch
import einops
import collections

class TriplaneNeRFRenderer(torch.nn.Module):
    def __init__(self, radius, valid_thresh, num_samples_per_ray):
        super().__init__()
        self.radius = radius #0.87 slightly larger than 0.5 * sqrt(3)
        self.valid_thresh = valid_thresh  #0.01
        self.feature_reduction = ["concat","mean"][0]
        self.density_activation = ["exp","trunc_exp"][0]
        self.density_bias = -1.0
        self.color_activation = "sigmoid"
        self.num_samples_per_ray = num_samples_per_ray
        self.chunk_size = 8192

    def query_triplane(self, decoder, positions, triplane):
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

        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)       
        positions = scale_tensor(positions, (-self.radius, self.radius), (-1, 1))  #positions in (-radius, radius), normalized to (-1, 1) for grid sample

        def chunk_batch(func, chunk_size, *args, **kwargs):
            if chunk_size <= 0:
                return func(*args, **kwargs)
            B = None
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, torch.Tensor):
                    B = arg.shape[0]
                    break
            assert (
                B is not None
            ), "No tensor found in args or kwargs, cannot determine batch size."
            out = collections.defaultdict(list)
            out_type = None
            # max(1, B) to support B == 0
            for i in range(0, max(1, B), chunk_size):
                out_chunk = func(
                    *[
                        arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ],
                    **{
                        k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                        for k, arg in kwargs.items()
                    },
                )
                if out_chunk is None:
                    continue
                out_type = type(out_chunk)
                if isinstance(out_chunk, torch.Tensor):
                    out_chunk = {0: out_chunk}
                elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
                    chunk_length = len(out_chunk)
                    out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
                elif isinstance(out_chunk, dict):
                    pass
                else:
                    print(
                        f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
                    )
                    exit(1)
                for k, v in out_chunk.items():
                    v = v if torch.is_grad_enabled() else v.detach()
                    out[k].append(v)

            if out_type is None:
                return None

            out_merged: Dict[Any, Optional[torch.Tensor]] = {}
            for k, v in out.items():
                if all([vv is None for vv in v]):
                    # allow None in return value
                    out_merged[k] = None
                elif all([isinstance(vv, torch.Tensor) for vv in v]):
                    out_merged[k] = torch.cat(v, dim=0)
                else:
                    raise TypeError(
                        f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
                    )

            if out_type is torch.Tensor:
                return out_merged[0]
            elif out_type in [tuple, list]:
                return out_type([out_merged[i] for i in range(chunk_length)])
            elif out_type is dict:
                return out_merged

        def _query_chunk(x):
            indices2D: torch.Tensor = torch.stack((x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]), dim=-3)
            out: torch.Tensor = torch.nn.functional.grid_sample(einops.rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3), einops.rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3), align_corners=False,mode="bilinear")
            if self.feature_reduction == "concat":
                out = einops.rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            elif self.feature_reduction == "mean":
                out = einops.reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")
            else:
                raise NotImplementedError
            net_out: Dict[str, torch.Tensor] = decoder(out)
            return net_out

        def get_activation(name):
            if name is None:
                return lambda x: x
            name = name.lower()
            if name == "none":
                return lambda x: x
            elif name == "exp":
                return lambda x: torch.exp(x)
            elif name == "sigmoid":
                return lambda x: torch.sigmoid(x)
            elif name == "tanh":
                return lambda x: torch.tanh(x)
            elif name == "softplus":
                return lambda x: F.softplus(x)
            else:
                try:
                    return getattr(F, name)
                except AttributeError:
                    raise ValueError(f"Unknown activation function: {name}")
        if self.chunk_size > 0:
            net_out = chunk_batch(_query_chunk, self.chunk_size, positions)
        else:
            net_out = _query_chunk(positions)
        net_out["density_act"] = get_activation(self.density_activation)(net_out["density"] + self.density_bias)
        net_out["color"] = get_activation(self.color_activation)(net_out["features"])
        net_out = {k: v.view(*input_shape, -1) for k, v in net_out.items()}
        return net_out

    def forward(self, decoder, triplane, rays_o, rays_d):  #, **kwargs
        def rays_intersect_bbox(
            rays_o: torch.Tensor,
            rays_d: torch.Tensor,
            radius: float,
            near: float = 0.0,
            valid_thresh: float = 0.01,
        ):
            input_shape = rays_o.shape[:-1]
            rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
            rays_d_valid = torch.where(
                rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d
            )
            if type(radius) in [int, float]:
                radius = torch.FloatTensor(
                    [[-radius, radius], [-radius, radius], [-radius, radius]]
                ).to(rays_o.device)
            radius = (
                1.0 - 1.0e-3
            ) * radius  # tighten the radius to make sure the intersection point lies in the bounding box
            interx0 = (radius[..., 1] - rays_o) / rays_d_valid
            interx1 = (radius[..., 0] - rays_o) / rays_d_valid
            t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
            t_far = torch.maximum(interx0, interx1).amin(dim=-1)

            # check wheter a ray intersects the bbox or not
            #t_diff = t_far - t_near
            #print('t_diff', t_diff.min(), t_diff.max())

            rays_valid = t_far - t_near > valid_thresh

            t_near[torch.where(~rays_valid)] = 0.0
            t_far[torch.where(~rays_valid)] = 0.0

            t_near = t_near.view(*input_shape, 1)
            t_far = t_far.view(*input_shape, 1)
            rays_valid = rays_valid.view(*input_shape)

            return t_near, t_far, rays_valid

        rays_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        n_rays = rays_o.shape[0]
        t_near, t_far, rays_valid = rays_intersect_bbox(rays_o, rays_d, self.radius, self.valid_thresh)

        t_near, t_far = t_near[rays_valid], t_far[rays_valid]
        t_vals = torch.linspace(0, 1, self.num_samples_per_ray + 1, device=triplane.device)
        t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
        z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]  # (N_rays, N_samples)
        #print(rays_o.shape, rays_d.shape, '  ', z_vals.shape)
        xyz = (rays_o[:, None, :] + z_vals[..., None] * rays_d[..., None, :])  # (N_rays, N_sample, 3)
        mlp_out = self.query_triplane(decoder=decoder, positions=xyz, triplane=triplane)
        eps = 1e-10
        deltas = t_vals[1:] - t_vals[:-1]  # (N_rays, N_samples)
        alpha = 1 - torch.exp(-deltas * mlp_out["density_act"][..., 0])  # (N_rays, N_samples)
        accum_prod = torch.cat([torch.ones_like(alpha[:, :1]), torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1)], dim=-1)
        weights = alpha * accum_prod  # (N_rays, N_samples)
        comp_rgb_ = (weights[..., None] * mlp_out["color"]).sum(dim=-2)  # (N_rays, 3)
        opacity_ = weights.sum(dim=-1)  # (N_rays)
        comp_rgb = torch.zeros(n_rays, 3, dtype=comp_rgb_.dtype, device=comp_rgb_.device)
        opacity = torch.zeros(n_rays, dtype=opacity_.dtype, device=opacity_.device)
        comp_rgb[rays_valid] = comp_rgb_
        opacity[rays_valid] = opacity_
        comp_rgb += 1 - opacity[..., None]
        comp_rgb = comp_rgb.view(*rays_shape, 3)
        opacity = opacity.view(*rays_shape, 1)
        return comp_rgb, opacity
