"""V-JEPA 2.1 ViT-B/16 (video, self-supervised) as the stage-2 refiner backbone.

ViTPose looks at the 32 frames of a refinement window one by one; V-JEPA 2.1 encodes the whole window as one
video (2-frame tubelets x 16x16 patches), so hand motion around the contact is seen directly. Weights: Meta's
`vjepa2_1_vitb_dist_vitG_384.pt` (ViT-B distilled from ViT-G), HF port pinned to a fixed revision.

refine.py is used unchanged: importing this module swaps refine.Refiner for a subclass that builds the usual
refiner (multi-level fusion, temporal pooling, heatmap head) and replaces only its backbone. The backbone
returns per-frame maps (each tubelet's features repeated for its 2 frames), so the rest of the refiner and the
in-span flag (one per frame) work as before.

train : python src/refine_vjepa.py train  <refine.py args> --backbone vjepa21-vitb --levels 5,8,11 ...
infer : python src/refine_vjepa.py submit <make_refined_submission.py args>
"""
import os, sys, runpy
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import refine

NAME = "vjepa21-vitb"
REPOS = {  # name -> (HF repo, pinned revision)
    "vjepa21-vitb": ("apiantonio/vjepa2.1-vit-base-384", "ea7765861aa689985c593727725afa378fc87492"),
    "vjepa21-vitl": ("apiantonio/vjepa2.1-vit-large-384", "d6cfdbdd818754f22eaa72e5320d97724765f099"),
    "vjepa21-vitg": ("apiantonio/vjepa2.1-vit-giant-384", "e04928c9b796b43f3797ad484de71ea33c031233"),
    "vjepa21-vitG": ("apiantonio/vjepa2.1-vit-gigantic-384", "ddfda36e2253b111f3a254060f5de35696d24c5e"),
}
REPO, REV = REPOS[NAME]


class VJepa21Features(nn.Module):
    class _Info:
        def __init__(self, ch): self.ch = ch
        def channels(self): return self.ch

    def __init__(self, layers, T, in_chans=4, name=NAME):
        super().__init__()
        from transformers import AutoModel
        repo, rev = REPOS[name]
        m = AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True)
        m.gradient_checkpointing_enable()
        enc = m.encoder
        old = enc.embeddings.patch_embeddings.proj
        new = nn.Conv3d(in_chans, old.out_channels, old.kernel_size, old.stride)
        with torch.no_grad():                         # 4th (hand heat) channel starts at zero
            new.weight.zero_(); new.weight[:, :3] = old.weight; new.bias.copy_(old.bias)
        enc.embeddings.patch_embeddings.proj = new
        enc.config.in_chans = in_chans
        self.enc, self.layers, self.T = enc, list(layers), T
        self.ts, self.p, C = enc.config.tubelet_size, enc.config.patch_size, enc.config.hidden_size
        self.feature_info = self._Info([C] * len(self.layers))

    def forward(self, x):                             # (B*T, 4, P, P), the T frames of each window in order
        BT, Cc, P, _ = x.shape
        T = self.T; B = BT // T; g = P // self.p; Tf = T // self.ts
        v = x.view(B, T, Cc, P, P).transpose(1, 2)    # (B, C, T, H, W)
        o = self.enc(v, out_layers=self.layers)
        out = []
        for h in o.multilevel_hidden_states:          # (B, Tf*g*g, C)
            C = h.shape[-1]
            h = h.view(B, Tf, g, g, C).permute(0, 1, 4, 2, 3)            # (B, Tf, C, g, g)
            out.append(h.repeat_interleave(self.ts, 1).reshape(B * T, C, g, g))
        return out


class RefinerVJ(refine.Refiner):
    def __init__(self, name="resnet18.a1_in1k", *a, levels=None, **kw):
        if name != NAME:
            return super().__init__(name, *a, levels=levels, **kw)
        # build the ViTPose-B refiner (same 3x768 fusion widths), then swap in the V-JEPA 2.1 backbone
        super().__init__("usyd-community/vitpose-plus-base", *a, levels=levels, **kw)
        self.net = VJepa21Features(levels, refine.T)


refine.Refiner = RefinerVJ

if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "train":
        sys.argv = ["refine.py"] + rest
        refine.main()
    elif mode == "submit":
        sys.argv = ["make_refined_submission.py"] + rest
        runpy.run_path(os.path.join(os.path.dirname(__file__), "make_refined_submission.py"), run_name="__main__")
    else:
        raise SystemExit("mode must be train or submit")
