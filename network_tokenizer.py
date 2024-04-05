import torch
import einops
import math

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        def drop_path(x, drop_prob: float = 0., training: bool = False):
            if drop_prob == 0. or not training:
                return x
            keep_prob = 1 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            output = x.div(keep_prob) * random_tensor
            return output
        return drop_path(x, self.drop_prob, self.training)

class Mlp(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(torch.nn.Module):  #https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, img_size=[224], patch_size=16, in_chans=3,embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=torch.nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        #self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))   #num_patches + 1
        self.pos_drop = torch.nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = torch.nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)
        #trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        #class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic', align_corners=True, recompute_scale_factor=True)    #John: align_corners=False  recompute_scale_factor=False
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        #return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return patch_pos_embed

    def prepare_tokens(self, x, interpolate_pos_encoding):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        #print('x', x.shape)
        if 0:  #John
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            #print('x', x.shape)
        p = self._interpolate_pos_encoding(x, w, h)
        #print('p', p.shape)
        x = x + p
        return self.pos_drop(x)

    def forward(self, x, interpolate_pos_encoding=True):
        x = self.prepare_tokens(x, interpolate_pos_encoding)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return dict(last_hidden_state=x, output=x[:, 0])

class DINOSingleImageTokenizer(torch.nn.Module):
    def __init__(self, img_size, depth, embed_dim, official):
        super().__init__()
        self.official = official
        if self.official:
            import transformers  #pip install transformers;  pip install accelerate
            self.model = transformers.models.vit.modeling_vit.ViTModel(transformers.models.vit.modeling_vit.ViTConfig(architectures=["ViTModel"], model_type="vit", torch_dtype="float32"))  #arguments all are default  #transformers.models.vit.modeling_vit.ViTModel.config_class.from_pretrained(config_file)
            #self.model.encoder.gradient_checkpointing = True
        else: 
            self.model = VisionTransformer(img_size=[img_size], in_chans=3, num_output=0, patch_size=16, embed_dim=embed_dim, num_heads=6, depth=depth, mlp_ratio=4, qkv_bias=True, norm_layer=torch.nn.LayerNorm)
            if 0:
                state_dict = torch.load(pretrain_file)  #https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth  https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth  https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.load_state_dict(state_dict, strict=True)
        self.register_buffer("image_mean", torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images, **kwargs):  #[1, 1, 3, 512, 512]
        if images.ndim == 4: images = images.unsqueeze(1)
        images = (images - self.image_mean) / self.image_std 
        imgs = einops.rearrange(images, "B N C H W -> (B N) C H W")  #[1, 3, 512, 512]

        if self.official:
            out = self.model(imgs, interpolate_pos_encoding=True).last_hidden_state  #[1, 3, 512, 512]  .last_hidden_state=[1, 1025, 768]  #hidden_size=768
        else:
            out = self.model.forward(imgs, interpolate_pos_encoding=True)['last_hidden_state']
            #print('out', out.shape)  #[1, 1025, 768]

        local_features = out.permute(0, 2, 1)
        local_features = einops.rearrange(local_features, "(B N) Ct Nt -> B N Ct Nt", B=images.shape[:1][0])
        if images.ndim == 4: local_features = local_features.squeeze(1)
        return local_features  #[1, 1, 768, 1025]

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError

import math
class Triplane1DTokenizer(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.plane_size = 32
        self.embeddings = torch.nn.Parameter(torch.randn((3, self.num_channels, self.plane_size, self.plane_size), dtype=torch.float32) * 1 / math.sqrt(self.num_channels))

    def forward(self, batch_size):
        return einops.rearrange(einops.repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size), "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")

    def detokenize(self, tokens):
        return einops.rearrange(tokens, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=self.plane_size, Wp=self.plane_size)
