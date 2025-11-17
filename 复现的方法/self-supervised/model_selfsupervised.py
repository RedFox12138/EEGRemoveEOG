"""
Self-Supervised EEG Denoising Model
复制自 Self-Supervised-EEG-Denoising-main/model.py
用于半模拟数据集
"""
import torch
from torch import nn
from einops import rearrange

class ConvEncoder(nn.Module):
    def __init__(self, dim, outdim, hidden_dim):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, outdim, kernel_size=1),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.conv(x)
   

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_q = nn.Conv1d(dim, hidden_dim, kernel_size=5, padding=5 // 2)
        self.to_k = nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=3 // 2)
        self.to_v = nn.Conv1d(dim, hidden_dim, kernel_size=1, padding=0)

        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        residual = x
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h=self.heads), (q, k, v))
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c x -> b (h c) x')
        return self.to_out(out) + residual
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.norm1 = nn.LayerNorm(length)
        self.attn = LinearAttention(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = ConvEncoder(in_channels, out_channels, in_channels * 2)
        self.conv2 = ConvEncoder(out_channels, out_channels, out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool1d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        layers = [
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip): 
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class DenoiseEEG(nn.Module):
    def __init__(self, in_channels=1, length = 512, n_feat = 128):
        super(DenoiseEEG, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, n_feat, 3, 1, 1), 
            nn.BatchNorm1d(n_feat),
            nn.GELU(),
        )

        self.down1 = UnetDown(n_feat, n_feat)
        self.trans1 = TransformerBlock(n_feat, length // 2)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.trans2 = TransformerBlock(n_feat * 2, length // 4)

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.trans3 = TransformerBlock(n_feat, length // 2)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.trans4 = TransformerBlock(n_feat, length)

        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((in_channels, None)),
            nn.Linear(length, length),
        )

    def forward(self, x):
        x = self.init_conv(x)

        down1 = self.down1(x)
        down1 = self.trans1(down1)

        down2 = self.down2(down1) 
        down2 = self.trans2(down2)

        up2 = self.up1(down2, down2)
        up2 = self.trans3(up2)

        up1 = self.up2(up2, down1)
        up1 = self.trans4(up1)

        out = self.out(torch.cat((up1, x), 1))
        return out
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
