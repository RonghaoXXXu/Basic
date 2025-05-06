import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image dimensions must be divisible by patch size ({self.patch_size})"
        
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x, (Hp, Wp)  # 返回 patch 后的高度和宽度用于后续 reshape

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), hidden_size, dropout=dropout)
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, temb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(temb).chunk(6, dim=1)

        # Attention
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]

        # MLP
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x)

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, temb):
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def get_timestep_embedding(timesteps, embedding_dim: int, max_period: int = 10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor"

    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:  # if odd, add zero padding
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.DiT.patch_size
        self.in_channels = config.DiT.in_chans* 2 if config.data.conditional else config.DiT.in_chans
        self.hidden_size = config.DiT.hidden_size
        self.depth = config.DiT.depth
        self.num_heads = config.DiT.num_heads
        self.mlp_ratio = config.DiT.mlp_ratio
        self.dropout = config.DiT.drop_rate

        # Timestep embedding
        self.temb = nn.Sequential(
            nn.Linear(config.DiT.t_ch, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, dropout=self.dropout)
            for _ in range(self.depth)
        ])

        # Final layer norm and projection
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size * self.patch_size * config.DiT.in_chans)

    def forward(self, x, t):
        # Timestep embedding
        temb = get_timestep_embedding(t, self.config.DiT.t_ch)
        temb = self.temb(temb)

        # Patch embedding
        x, (Hp, Wp) = self.patch_embed(x)  # x: [B, N, D]

        # Apply transformer blocks with conditioning via AdaLN
        for block in self.blocks:
            x = block(x, temb)

        # Final projection to image space
        x = self.final_layer(x, temb)

        # Reshape back to image format
        B, _, C = x.shape
        x = x.reshape(B, Hp, Wp, self.patch_size, self.patch_size, C // (self.patch_size ** 2))
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * self.patch_size, Wp * self.patch_size)

        return x


if __name__ == '__main__':
    from argparse import Namespace
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    input_high = torch.randn(2, 3, 256, 256).to(device)
    gt_high = torch.randn(2, 3, 256, 256).to(device)
    num_timesteps = 200
    t = torch.randint(low=0, high=num_timesteps, size=(input_high.shape[0] // 2 + 1,)).to(input_high.device)
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:input_high.shape[0]].to(input_high.device)

    from omegaconf import OmegaConf
    config = OmegaConf.load('/datadisk2/xuronghao/Projects/Basic/configs/ll.yml')

    model = DiT(config).to(device)

    output = model(torch.cat([input_high, gt_high], dim=1), t)
    print(output.shape)