"""True 3D video backbones (Kinetics-pretrained) -> (B, T', S, C) tokens.

Why this exists: the per-frame 2D backbone has no temporal receptive field, and
the measured weak spot of run v1 is temporal localization (mAPt < mAPs). These
backbones model time directly.

Resolution note: at the median scale of 3.19 m per normalized unit, a 7x7 token
grid is 0.456 m per cell - coarser than the metric's strictest spatial
thresholds (0.1-0.2 m). Spatial precision therefore cannot come from the grid;
it comes from the pose anchor (1/32 grid = 0.10 m) plus the regressed offset.
The optional high-resolution stage below gives that offset a finer substrate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


class Swin3DBackbone(nn.Module):
    """Video Swin-T. Returns the final stage, optionally fused with an earlier,
    higher-resolution stage."""

    def __init__(self, size="t", pretrained=True, hires=True, out_frames=16):
        super().__init__()
        from torchvision.models import video as V
        builder = {"t": V.swin3d_t, "s": V.swin3d_s, "b": V.swin3d_b}[size]
        weights = {"t": V.Swin3D_T_Weights, "s": V.Swin3D_S_Weights,
                   "b": V.Swin3D_B_Weights}[size].KINETICS400_V1 if pretrained else None
        m = builder(weights=weights)
        self.patch_embed = m.patch_embed
        self.pos_drop = m.pos_drop
        self.features = m.features
        self.hires = hires
        self.out_frames = out_frames
        # channel dims per torchvision's layout: features = [stage1, down, stage2, ...]
        self.feat_dim = m.norm.normalized_shape[0]
        # torchvision layout: [stage1, merge, stage2, merge, stage3, merge, stage4]
        # features[-3] is the 14x14 stage at half the final channel width.
        self.mid_idx = len(self.features) - 3
        self.mid_dim = self.feat_dim // 2
        if hires:
            self.lateral = nn.Linear(self.mid_dim, self.feat_dim)
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def forward(self, clips):                      # (B,T,H,W,3) uint8
        B, T, H, W, _ = clips.shape
        x = clips.permute(0, 4, 1, 2, 3).float().div_(255)     # (B,3,T,H,W)
        x = (x - self.mean) / self.std
        x = self.pos_drop(self.patch_embed(x))                 # (B,T',h,w,C)
        mid = None
        for i, blk in enumerate(self.features):
            x = blk(x)
            if self.hires and i == self.mid_idx:
                mid = x
        # x: (B, t, h, w, C)
        b, t, h, w, c = x.shape
        out = x.reshape(b, t, h * w, c)
        if self.hires and mid is not None:
            mb, mt, mh, mw, _ = mid.shape
            m = self.lateral(mid).reshape(mb, mt, mh * mw, c)
            # concat along the token axis; both grids keep their own resolution
            out = torch.cat([out, m], dim=2)
        return out


def build_video_backbone(name="swin3d_t", pretrained=True, img_size=224,
                         num_frames=16, hires=True):
    size = name.split("_")[-1][-1] if name.startswith("swin3d") else "t"
    bb = Swin3DBackbone(size, pretrained, hires, num_frames)
    with torch.no_grad():
        d = bb(torch.zeros(1, num_frames, img_size, img_size, 3, dtype=torch.uint8))
    return bb, bb.feat_dim, d.shape[2], d.shape[1]
