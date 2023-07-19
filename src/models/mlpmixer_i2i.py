import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b h w c")
        )

    def forward(self, x):
        return self.proj(x)


class PatchExpansion(nn.Module):
    def __init__(self, dim_scale, embed_dim, out_chans):
        super().__init__()
        self.dim_scale = dim_scale
        self.output_dim = embed_dim 
        self.expand = nn.Linear(embed_dim, dim_scale**2* embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)
        x = x.view(B,H*self.dim_scale, W*self.dim_scale,-1)
        x = x.permute(0,3,1,2)
        x = self.output(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.model(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches_h, num_patches_w, f_hidden, drop=0.):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b h w c -> b c w h"),
            Mlp(num_patches_h, num_patches_h*f_hidden, drop=drop),
            Rearrange("b c w h -> b c h w"),
            Mlp(num_patches_w, num_patches_w*f_hidden, drop=drop),
            Rearrange("b c h w -> b h w c"),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, dim*f_hidden, drop=drop)
        )
    
    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class MixerNet(nn.Module):
    def __init__(
                self,
                img_size_in=(992, 1344),
                img_size_out=(512, 512), 
                patch_size=(16, 16), 
                in_chans=6, 
                out_chans=1,
                embed_dim=512, 
                num_blocks=16,
                f_hidden=4, 
                drop_rate=0.,
                **kwargs
    ):
        super().__init__()
        num_patches_h = img_size_in[0] // patch_size[0]
        num_patches_w = img_size_in[1] // patch_size[1]
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        layers = [ 
            MixerBlock(embed_dim, num_patches_h, num_patches_w, f_hidden, drop=drop_rate)
            for _ in range(num_blocks)]
        self.mixer_layers = nn.Sequential(*layers)
        self.patch_expand = PatchExpansion(patch_size[0], embed_dim, out_chans)
        self.resize = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.mixer_layers(x)
        x = self.patch_expand(x)
        x = self.resize(x)
        return x
    