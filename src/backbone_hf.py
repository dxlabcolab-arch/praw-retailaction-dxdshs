"""Video backbones pretrained on data other than Kinetics, via Hugging Face.

  * videomae_ssv2 : VideoMAE ViT-B fine-tuned on Something-Something V2 - hand-object
                    interactions ("picking something up", "putting something ...").
                    License CC-BY-NC-4.0.
  * vjepa2_l      : V-JEPA 2 ViT-L, self-supervised on 1M+ hours of video (incl. SSv2).
                    License MIT. Native resolution 256.

Both return (B, T', S, C) spatio-temporal tokens like the other backbones.
"""
import torch
import torch.nn as nn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _load_videomae(repo, num_frames):
    """VideoMAE with the original q_bias/v_bias mapped onto query/value biases.
    Recent transformers versions no longer convert them and leave the biases random."""
    from transformers import VideoMAEConfig, VideoMAEModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import os
    cfg = VideoMAEConfig.from_pretrained(repo)
    cfg.num_frames = num_frames                  # sinusoidal position table, rebuilt for T
    m = VideoMAEModel(cfg)
    snap = os.path.dirname(hf_hub_download(repo, "config.json"))
    f = os.path.join(snap, "model.safetensors")
    sd = load_file(f) if os.path.exists(f) else torch.load(os.path.join(snap, "pytorch_model.bin"), map_location="cpu")
    new = {}
    for k, v in sd.items():
        k = k.removeprefix("videomae.")
        if k.endswith("attention.attention.q_bias"):
            base = k[: -len("q_bias")]
            new[base + "query.bias"] = v
            new[base + "key.bias"] = torch.zeros_like(v)
        elif k.endswith("attention.attention.v_bias"):
            new[k[: -len("v_bias")] + "value.bias"] = v
        else:
            new[k] = v
    missing, unexpected = m.load_state_dict(new, strict=False)
    missing = [k for k in missing if "position" not in k]
    assert not missing, f"VideoMAE weights missing: {missing[:5]}"
    return m, cfg.hidden_size, cfg.image_size // cfg.patch_size, cfg.tubelet_size


class HFVideoBackbone(nn.Module):
    def __init__(self, name, num_frames=32, grad_ckpt=True):
        super().__init__()
        self.name = name
        if name == "videomae_ssv2":
            self.net, self.feat_dim, self.grid, self.tub = _load_videomae(
                "MCG-NJU/videomae-base-finetuned-ssv2", num_frames)
            self.norm = nn.LayerNorm(self.feat_dim)      # checkpoint used mean-pooling, no final LN
        elif name in ("vjepa2_l", "vjepa2_g"):
            from transformers import AutoModel
            self.net = AutoModel.from_pretrained(
                {"vjepa2_l": "facebook/vjepa2-vitl-fpc64-256", "vjepa2_g": "facebook/vjepa2-vitg-fpc64-256"}[name])
            c = self.net.config
            self.feat_dim, self.grid, self.tub = c.hidden_size, c.crop_size // c.patch_size, c.tubelet_size
            self.norm = nn.Identity()
        else:
            raise ValueError(name)
        if grad_ckpt and hasattr(self.net, "gradient_checkpointing_enable"):
            self.net.gradient_checkpointing_enable()
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)

    def forward(self, clips):                                   # (B,T,H,W,3) uint8
        B, T = clips.shape[:2]
        x = clips.permute(0, 1, 4, 2, 3).float().div_(255)      # (B,T,3,H,W)
        x = (x - self.mean) / self.std
        if self.name.startswith("vjepa2"):
            h = self.net(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        else:
            h = self.net(pixel_values=x).last_hidden_state
        h = self.norm(h)
        return h.view(B, T // self.tub, self.grid * self.grid, self.feat_dim)


def build_hf_backbone(name, num_frames=32):
    bb = HFVideoBackbone(name, num_frames)
    return bb, bb.feat_dim, bb.grid * bb.grid
