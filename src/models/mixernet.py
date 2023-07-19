import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(512, 512), patch_size=(16, 16), in_chans=1, embed_dim=512):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixing = Mlp(num_patches, tokens_mlp_dim, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixing = Mlp(dim, channels_mlp_dim, drop=drop)

    def forward(self, x):
        out = self.norm1(x).transpose(1, 2)
        x = x + self.token_mixing(out).transpose(1, 2)
        out = self.norm2(x)
        return x + self.channel_mixing(out)


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
        self.img_size_in = img_size_in
        self.img_size_out = img_size_out
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size=img_size_in, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.blocks = nn.ModuleList([
            MixerBlock(embed_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, drop=drop_rate)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)
        self.resize = nn.Upsample(size=img_size_out, mode='bilinear', align_corners=True)

        trunc_normal_(self.head.weight, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size_in[0] // self.patch_size[0],
            w=self.img_size_in[1] // self.patch_size[1],
        )
        x = self.resize(x)
        return x
    