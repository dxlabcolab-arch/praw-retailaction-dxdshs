"""The dataset paper's baseline head, as described in Sec. 6 of Mazzini et al.:

  shared backbone -> linear projection + learned 3D positional encoding
  -> per-view transformer encoder -> concat + view embedding
  -> DETR decoder over learned action queries
  -> class logits, [start, end], and a free (x, y) per view.

No pose input, no anchor, no frame branch: this is the reference point for
measuring what pose anchoring (model.py) and FACT (model_fact.py) add.
"""
import torch
import torch.nn as nn

from model import MultiViewPoseDETR, mlp


class PlainMultiViewDETR(MultiViewPoseDETR):
    def __init__(self, backbone, feat_dim, dim=256, num_queries=5, num_classes=3,
                 num_frames=16, tokens_per_frame=49, dec_layers=4, enc_layers=2):
        super().__init__(backbone, feat_dim, dim, num_queries, num_classes,
                         num_frames, tokens_per_frame, dec_layers, enc_layers)
        # free point regression, as in the paper
        self.point_head = mlp(dim, dim, 4, 3)
        # unused inherited parts: pose encoder and anchor/offset heads
        self.pose_enc = None
        self.anchor_head = self.offset_head = self.anchor_t_head = None

    def _tokens(self, clips, poses=None):
        B, V, T = clips.shape[:3]
        feats = self.backbone(clips.flatten(0, 1))
        f = self.proj(feats)
        BV, Tf, S, D = f.shape
        g = self.grid
        pos = (self.pos_t[:Tf].view(Tf, 1, D)
               + (self.pos_h.view(g, 1, D) + self.pos_w.view(1, g, D)).view(1, g * g, D)[:, :S])
        f = self.encoder((f + pos).flatten(1, 2))
        f = f.view(B, V, -1, D) + self.view_emb.view(1, V, 1, D)
        return f.flatten(1, 2)

    def forward(self, clips, poses=None):
        B = clips.shape[0]
        mem = self._tokens(clips)
        h = self.decoder(self.query.unsqueeze(0).expand(B, -1, -1), mem)
        logits = self.cls_head(h)
        span = self.span_head(h).sigmoid()
        center, width = span[..., 0], span[..., 1]
        pts = self.point_head(h).sigmoid()                                  # (B,Q,4)
        return {"logits": logits,
                "start": (center - width / 2).clamp(0, 1),
                "end": (center + width / 2).clamp(0, 1),
                "points": pts}
