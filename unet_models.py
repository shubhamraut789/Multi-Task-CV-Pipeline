"""
unet_models.py  –  Three U-Net variants for Question 2
=======================================================

2.1  VanillaMultiTaskUNet      – standard double-conv encoder-decoder with skip
2.2  NoSkipMultiTaskUNet       – same architecture, skip connections removed
2.3  ResidualMultiTaskUNet     – residual blocks in encoder and decoder

All three models:
  Input  : [B, 3, 256, 256]  RGB image
  Output : seg   [B, 14, 256, 256]  segmentation logits
           depth [B,  1, 256, 256]  depth map (sigmoid → [0,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  Vanilla U-Net building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two consecutive  Conv-BN-ReLU  blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Strided MaxPool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample + optional skip concat + DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        cat_ch = in_ch + skip_ch if use_skip else in_ch
        self.conv = DoubleConv(cat_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip and skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  Vanilla Multi-Task U-Net  (with skip connections)
# ═══════════════════════════════════════════════════════════════════════════════

class VanillaMultiTaskUNet(nn.Module):
    """
    Encoder: 3→64→128→256→512→1024   (strided MaxPool between stages)
    Decoder: shared until final stage, then split into seg head + depth head
    Skip connections: yes (concat encoder features with decoder features)
    """
    def __init__(self, num_classes=14):
        super().__init__()
        # ── encoder ───────────────────────────────────────────────────────────
        self.enc1 = DoubleConv(3,    64)     # 256 → 256
        self.enc2 = Down(64,  128)           # 256 → 128
        self.enc3 = Down(128, 256)           # 128 →  64
        self.enc4 = Down(256, 512)           #  64 →  32
        self.enc5 = Down(512, 1024)          #  32 →  16  (bottleneck)

        # ── decoder (shared) ──────────────────────────────────────────────────
        self.dec4 = Up(1024, 512, 512)       # 16  →  32
        self.dec3 = Up(512,  256, 256)       # 32  →  64
        self.dec2 = Up(256,  128, 128)       # 64  → 128
        self.dec1 = Up(128,   64,  64)       # 128 → 256

        # ── output heads ──────────────────────────────────────────────────────
        self.seg_head   = nn.Conv2d(64, num_classes, 1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # [B, 64, 256, 256]
        e2 = self.enc2(e1)   # [B,128, 128, 128]
        e3 = self.enc3(e2)   # [B,256,  64,  64]
        e4 = self.enc4(e3)   # [B,512,  32,  32]
        e5 = self.enc5(e4)   # [B,1024, 16,  16]

        # Decoder with skip connections
        d4 = self.dec4(e5, e4)  # [B,512, 32,32]
        d3 = self.dec3(d4, e3)  # [B,256, 64,64]
        d2 = self.dec2(d3, e2)  # [B,128,128,128]
        d1 = self.dec1(d2, e1)  # [B, 64,256,256]

        seg   = self.seg_head(d1)    # [B,14,256,256]
        depth = self.depth_head(d1)  # [B, 1,256,256]
        return seg, depth


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2  Multi-Task U-Net WITHOUT skip connections
# ═══════════════════════════════════════════════════════════════════════════════

class NoSkipMultiTaskUNet(nn.Module):
    """
    Identical channel config as VanillaMultiTaskUNet but skip=False everywhere.
    Decoder input channels are halved because there is no skip concat.
    """
    def __init__(self, num_classes=14):
        super().__init__()
        # ── encoder (same as vanilla) ─────────────────────────────────────────
        self.enc1 = DoubleConv(3,    64)
        self.enc2 = Down(64,  128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        self.enc5 = Down(512, 1024)

        # ── decoder WITHOUT skip concat ───────────────────────────────────────
        # in_ch = encoder output, skip_ch=0 so cat_ch == in_ch
        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(1024, 512)
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(512,  256)
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(256,  128)
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(128,   64)

        # ── output heads ──────────────────────────────────────────────────────
        self.seg_head   = nn.Conv2d(64, num_classes, 1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder — NO skip connections
        d4 = self.dec4(self.up4(e5))
        d3 = self.dec3(self.up3(d4))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        seg   = self.seg_head(d1)
        depth = self.depth_head(d1)
        return seg, depth


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3  Multi-Task U-Net with Residual Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Two 3×3 Conv-BN-ReLU layers with a residual (identity or 1×1 proj) shortcut.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # 1×1 projection if channel dimensions differ
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResDown(nn.Module):
    """MaxPool + ResidualBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ResUp(nn.Module):
    """Bilinear upsample + skip concat + ResidualBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualMultiTaskUNet(nn.Module):
    """
    Same topology as VanillaMultiTaskUNet but every DoubleConv replaced by
    ResidualBlock. Skip connections retained.
    """
    def __init__(self, num_classes=14):
        super().__init__()
        # ── encoder ───────────────────────────────────────────────────────────
        self.enc1 = ResidualBlock(3,    64)
        self.enc2 = ResDown(64,  128)
        self.enc3 = ResDown(128, 256)
        self.enc4 = ResDown(256, 512)
        self.enc5 = ResDown(512, 1024)

        # ── decoder ───────────────────────────────────────────────────────────
        self.dec4 = ResUp(1024, 512, 512)
        self.dec3 = ResUp(512,  256, 256)
        self.dec2 = ResUp(256,  128, 128)
        self.dec1 = ResUp(128,   64,  64)

        # ── output heads ──────────────────────────────────────────────────────
        self.seg_head   = nn.Conv2d(64, num_classes, 1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        seg   = self.seg_head(d1)
        depth = self.depth_head(d1)
        return seg, depth