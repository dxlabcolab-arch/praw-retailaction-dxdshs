"""Video backbone wrappers returning (B, T', S, C) spatio-temporal tokens."""
import torch
import torch.nn as nn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TimmFrameBackbone(nn.Module):
    """Per-frame 2D backbone (e.g. ViT / ConvNeXt) applied across time.

    Cheap, strong ImageNet/CLIP init, and keeps a real spatial grid - which the
    metric cares about, since it scores point localization.
    """

    def __init__(self, name="convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True,
                 img_size=224):
        super().__init__()
        import timm
        self.net = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(-1,)
        )
        self.feat_dim = self.net.feature_info.channels()[-1]
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def forward(self, clips):          # (B,T,H,W,3) uint8
        B, T, H, W, _ = clips.shape
        x = clips.permute(0, 1, 4, 2, 3).reshape(B * T, 3, H, W).float().div_(255)
        x = (x - self.mean) / self.std
        f = self.net(x)[-1]                       # (B*T, C, h, w)
        C, h, w = f.shape[1:]
        f = f.flatten(2).transpose(1, 2)          # (B*T, h*w, C)
        return f.view(B, T, h * w, C)


def build_backbone(name, pretrained=True, img_size=224):
    bb = TimmFrameBackbone(name, pretrained, img_size)
    with torch.no_grad():
        d = bb(torch.zeros(1, 2, img_size, img_size, 3, dtype=torch.uint8))
    return bb, bb.feat_dim, d.shape[2]
