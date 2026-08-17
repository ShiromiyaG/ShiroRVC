import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    # Lightweight attention mechanism.
    # Pools features along Frequency (H) and Time (W) separately.
    # This helps the model capture long-range dependencies in both directions.
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # Pool Frequency
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # Pool Time

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv1d(in_channels, mip, kernel_size=1, bias=False)

        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=mip)

        self.act = nn.GELU()

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
        self.gelu = nn.GELU()

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
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.attn(out) # Apply Coordinate Attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)
        return out

class DSCA_ResUNet_v3(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(DSCA_ResUNet_v3, self).__init__()

        # Initial Feature Extraction
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU()
        )

        # Encoder Path:
        # Note the asymmetric strides later on (2, 1) which compress Frequency more than Time.
        self.enc1 = ResBlock(32, 64, stride=2)   
        self.enc2 = ResBlock(64, 128, stride=2)  

        self.enc3 = ResBlock(128, 256, stride=(2, 1)) 
        self.enc4 = ResBlock(256, 512, stride=(2, 1)) 

        # Bridge (Bottleneck)
        self.bridge = ResBlock(512, 512, stride=1) 

        # Decoder Path:
        # Upsampling and concatenation (skip connections) to recover spatial details.

        # Up 4
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.GELU()
        )
        self.dec4 = ResBlock(512, 256)

        # Up 3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU()
        )
        self.dec3 = ResBlock(256, 128)

        # Up 2: Symmetric (2, 2) -> Use PixelShuffle
        self.up2_conv = nn.Conv2d(128, 64 * 4, kernel_size=1)
        self.up2_ps = nn.PixelShuffle(2)
        self.dec2 = ResBlock(128, 64)

        # Up 1
        self.up1_conv = nn.Conv2d(64, 32 * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(2)
        self.dec1 = ResBlock(64, 32)

        # Deep Supervision Heads
        self.ds_out2 = nn.Conv2d(64, n_classes, kernel_size=1)  # At Dec 2
        self.ds_out3 = nn.Conv2d(128, n_classes, kernel_size=1) # At Dec 3

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
        d4 = torch.cat([e3, d4], dim=1)
        d4 = self.dec4(d4)

        # Decoder 3
        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]: 
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='nearest')
        d3 = torch.cat([e2, d3], dim=1)
        d3 = self.dec3(d3)

        # Decoder 2
        d2 = self.up2_conv(d3)
        d2 = self.up2_ps(d2)
        if d2.shape[2:] != e1.shape[2:]: 
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='nearest')
        d2 = torch.cat([e1, d2], dim=1)
        d2 = self.dec2(d2)

        # Decoder 1
        d1 = self.up1_conv(d2)
        d1 = self.up1_ps(d1)
        if d1.shape[2:] != x1.shape[2:]: 
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='nearest')
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        logits = self.outc(d1)

        if self.training:
            # If training, return main output + aux outputs for deep supervision loss.
            aux2 = self.ds_out2(d2)
            aux3 = self.ds_out3(d3)

            # interpolation for aux outputs
            aux2 = F.interpolate(aux2, size=logits.shape[2:], mode='bilinear', align_corners=True)
            aux3 = F.interpolate(aux3, size=logits.shape[2:], mode='bilinear', align_corners=True)

            return { "main": logits, "aux2": aux2, "aux3": aux3 }
        else:
            return logits
