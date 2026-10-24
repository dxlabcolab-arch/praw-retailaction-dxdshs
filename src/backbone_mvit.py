"""MViT-v2-S (torchvision, Kinetics-400) -> (B, T', S, C) spatio-temporal tokens.

This is the backbone family the dataset paper's best baseline used (MViT-b, 41.7
mAP in the paper's own grid). torchvision ships the small variant; with 16 input
frames it yields 8 temporal slots x 7x7 spatial tokens x 768 channels, the same
layout as Swin3D-T, so the rest of the model is unchanged.
"""
import torch
import torch.nn as nn

# torchvision video-model normalization
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)


class MViTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None)
        self.conv_proj, self.pos_encoding, self.blocks, self.norm = m.conv_proj, m.pos_encoding, m.blocks, m.norm
        self.feat_dim = m.norm.normalized_shape[0]
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def forward(self, clips):                                   # (B,T,H,W,3) uint8
        B = clips.shape[0]
        x = clips.permute(0, 4, 1, 2, 3).float().div_(255)      # (B,3,T,H,W)
        x = (x - self.mean) / self.std
        x = self.conv_proj(x).flatten(2).transpose(1, 2)        # (B, T'*H'*W', C)
        x = self.pos_encoding(x)                                # prepends the class token
        thw = (self.pos_encoding.temporal_size,) + tuple(self.pos_encoding.spatial_size)
        for blk in self.blocks:
            x, thw = blk(x, thw)
        x = self.norm(x)[:, 1:]                                 # drop class token
        t, h, w = thw
        return x.reshape(B, t, h * w, self.feat_dim)


def build_mvit_backbone(pretrained=True, img_size=224, num_frames=16):
    bb = MViTBackbone(pretrained)
    with torch.no_grad():
        d = bb(torch.zeros(1, num_frames, img_size, img_size, 3, dtype=torch.uint8))
    return bb, bb.feat_dim, d.shape[2]
