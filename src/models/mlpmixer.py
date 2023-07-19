import torch.nn as nn
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x):
        return self.proj(x)


class PatchEmbeddings_transpose(nn.Module):
    def __init__(self, patch_size, out_chans, embed_dim, h):
        super().__init__()
        self.proj_transpose = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=h),
            nn.Conv2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_size)
        )
    
    def forward(self, x):
        return self.proj_transpose(x)


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
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=0.):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b p c -> b c p"),
            Mlp(num_patches, tokens_mlp_dim),
            Rearrange("b c p -> b p c")
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(dim, channels_mlp_dim)
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
                tokens_mlp_dim=256, 
                channels_mlp_dim=2048,
                num_blocks=8, 
                drop_rate=0., 
                **kwargs):
        super().__init__()
        num_patches = (img_size_in[1] // patch_size[1]) * (img_size_in[0] // patch_size[0])
        num_patches_h = img_size_in[0] // patch_size[0]
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        layers = [ 
            MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
            for _ in range(num_blocks)]
        self.mixer_layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_transpose = PatchEmbeddings_transpose(patch_size, out_chans, embed_dim, num_patches_h)
        self.resize = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.patch_embed(x)           
        x = self.mixer_layers(x)           
        x = self.norm(x)            
        x = self.embed_transpose(x)
        x = self.resize(x)
        return x  
    