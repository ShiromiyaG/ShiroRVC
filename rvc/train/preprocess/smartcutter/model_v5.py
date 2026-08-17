import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

        self.silu = nn.SiLU(inplace=True)

    def forward(self, g, x):
        # g: Gating Signal (from Decoder)
        # x: Skip Connection (from Encoder)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.silu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class CoordinateAttention(nn.Module):
    # Lightweight attention mechanism.
    # Pools features along Frequency (H) and Time (W) separately.
    # This helps the model capture long-range dependencies in both directions.
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Pool Frequency
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Pool Time

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv1d(in_channels, mip, kernel_size=7, stride=1, padding=3, bias=False)

        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=mip)

        self.act = nn.SiLU()

        self.conv_h = nn.Conv1d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv1d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool separately then concatenate to process spatial info together.
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y.squeeze(-1))
        y = self.gn1(y)
        y = self.act(y) 

        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Compute attention maps for H and W axes.
        a_h = self.conv_h(x_h).sigmoid().unsqueeze(-1)
        a_w = self.conv_w(x_w).sigmoid().unsqueeze(-1).permute(0, 1, 3, 2)

        return identity * a_h * a_w


class ResBlock(nn.Module):
    # Standard Residual Block enhanced with Coordinate Attention.
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.stride = stride 

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # The attention layer added at the end of the block.
        self.attn = CoordinateAttention(out_channels)
        self.silu = nn.SiLU(inplace=True)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.attn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.silu(out)
        return out


class DilatedBridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Parallel dilated paths (like ASPP)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)

        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_channels * 3)

        self.out_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x4 = self.conv4(x)
        out = torch.cat([x1, x2, x4], dim=1)
        return self.silu(self.out_conv(self.gn(out)))


class CGA_ResUNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(CGA_ResUNet, self).__init__()

        # Initial Feature Extraction ( encoder )
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(inplace=True)
        )

        # Encoder Path
        self.enc1 = ResBlock(32, 64, stride=2)   
        self.enc2 = ResBlock(64, 128, stride=2)  
        self.enc3 = ResBlock(128, 256, stride=(2, 1)) 
        self.enc4 = ResBlock(256, 512, stride=(2, 1)) 

        # Bridge (Bottleneck)
        self.bridge = DilatedBridge(512, 512)

        # Decoder with attention gates
        # Up 4
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(inplace=True)
        )
        # Gate 4: Filters 'enc3' (256 ch) using 'up4' (256 ch)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = ResBlock(512, 256)

        # Up 3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(inplace=True)
        )
        # Gate 3: Filters 'enc2' (128 ch) using 'up3' (128 ch)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ResBlock(256, 128)

        # Up 2: Symmetric (2, 2) -> Use PixelShuffle
        self.up2_conv = nn.Conv2d(128, 64 * 4, kernel_size=1)
        self.up2_ps = nn.PixelShuffle(2)

        # Gate 2: Filters 'enc1' (64 ch) using 'up2' (64 ch)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = ResBlock(128, 64)

        # Up 1
        self.up1_conv = nn.Conv2d(64, 32 * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(2)
        # Gate 1: Filters 'x1' (32 ch) using 'up1' (32 ch)
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = ResBlock(64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        e1 = self.enc1(x1) 
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        e4 = self.enc4(e3)

        b = self.bridge(e4)

        # Decoder 4
        d4 = self.up4(b)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode='nearest')

        # Attention gate 4
        x4_gated = self.att4(g=d4, x=e3)
        d4 = torch.cat([x4_gated, d4], dim=1)

        d4 = self.dec4(d4)

        # Decoder 3
        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='nearest')

        # Attention gate 3
        x3_gated = self.att3(g=d3, x=e2)
        d3 = torch.cat([x3_gated, d3], dim=1)

        d3 = self.dec3(d3)

        # Decoder 2
        d2 = self.up2_conv(d3)
        d2 = self.up2_ps(d2)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='nearest')

        # Attention gate 2
        x2_gated = self.att2(g=d2, x=e1)
        d2 = torch.cat([x2_gated, d2], dim=1)

        d2 = self.dec2(d2)

        # Decoder 1
        d1 = self.up1_conv(d2)
        d1 = self.up1_ps(d1)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='nearest')

        # Attention gate 1
        x1_gated = self.att1(g=d1, x=x1)
        d1 = torch.cat([x1_gated, d1], dim=1)

        d1 = self.dec1(d1)

        # Output
        logits = self.outc(d1)

        return logits