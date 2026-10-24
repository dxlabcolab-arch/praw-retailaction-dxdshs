"""Pose-anchored multi-view DETR for RetailAction.

Design rationale (from data analysis):
 * GT interaction points sit near the hand at the *reach apex*; hand keypoints
   give a strong prior but are quantized on a 1/32 grid (~0.10 m of error at the
   median scale), so they cannot be copied as answers. We therefore let pose
   provide ANCHORS and regress a visual residual on top.
 * Matching accepts a prediction if it is close in EITHER view, so the two view
   heads must be able to disagree; we keep per-view spatial heads.
 * 97% of actions are `take`, so ranking quality (score calibration) and spatial
   precision dominate the metric far more than classification.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_JOINTS, HAND_JOINTS, JOINT_NAMES

HAND_IDX = [JOINT_NAMES.index(j) for j in HAND_JOINTS]


def mlp(i, h, o, n=3, drop=0.0):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        if drop:
            layers += [nn.Dropout(drop)]
        d = h
    layers += [nn.Linear(d, o)]
    return nn.Sequential(*layers)


class PoseEncoder(nn.Module):
    """Encodes the per-view pose sequence into tokens (one per frame)."""

    def __init__(self, dim, num_frames):
        super().__init__()
        self.joint_emb = nn.Parameter(torch.randn(NUM_JOINTS, dim) * 0.02)
        self.inp = nn.Linear(3, dim)
        self.temporal = nn.Parameter(torch.randn(num_frames, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            dim, 4, dim * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc, 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, pose):                       # (B,T,J,3)
        B, T, J, _ = pose.shape
        # velocity carries the reach dynamics that mark the apex
        vel = torch.zeros_like(pose[..., :2])
        vel[:, 1:] = pose[:, 1:, :, :2] - pose[:, :-1, :, :2]
        x = self.inp(pose) + self.joint_emb                      # (B,T,J,D)
        x = x + self.inp(torch.cat([vel, pose[..., 2:]], -1))
        x = x.mean(2) + self.temporal[:T]                        # (B,T,D)
        return self.norm(self.tr(x))


class MultiViewPoseDETR(nn.Module):
    def __init__(self, backbone, feat_dim, dim=256, num_queries=10,
                 num_classes=3, num_frames=16, tokens_per_frame=49,
                 dec_layers=4, enc_layers=2):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.dim = dim

        self.proj = nn.Linear(feat_dim, dim)
        # learned 3D positional encoding: (t, h, w) factorized
        self.pos_t = nn.Parameter(torch.randn(num_frames, dim) * 0.02)
        g = int(round(math.sqrt(tokens_per_frame)))
        self.grid = g
        self.pos_h = nn.Parameter(torch.randn(g, dim) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(g, dim) * 0.02)
        self.view_emb = nn.Parameter(torch.randn(2, dim) * 0.02)

        enc = nn.TransformerEncoderLayer(
            dim, 8, dim * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, enc_layers)

        self.pose_enc = PoseEncoder(dim, num_frames)

        dec = nn.TransformerDecoderLayer(
            dim, 8, dim * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.cls_head = mlp(dim, dim, num_classes + 1, 2)     # +1 = "no action"
        self.span_head = mlp(dim, dim, 2, 3)
        # per-view: pick an anchor among hand joints/frames, then refine
        self.anchor_head = mlp(dim, dim, 2 * len(HAND_IDX), 2)   # logits per view
        self.offset_head = mlp(dim, dim, 2 * 2, 3)               # (dx,dy) per view
        self.anchor_t_head = mlp(dim, dim, 2 * num_frames, 2)    # which frame, per view

    def _tokens(self, clips, poses):
        """clips (B,2,T,H,W,3)->backbone; returns memory (B, L, D)."""
        B, V, T = clips.shape[:3]
        feats = self.backbone(clips.flatten(0, 1))     # (B*V, T', S, feat)
        f = self.proj(feats)
        BV, Tf, S, D = f.shape
        g = self.grid
        pos = (
            self.pos_t[:Tf].view(Tf, 1, D)
            + (self.pos_h.view(g, 1, D) + self.pos_w.view(1, g, D)).view(1, g * g, D)[:, :S]
        )
        f = f + pos
        f = f.flatten(1, 2)                            # (B*V, Tf*S, D)
        f = self.encoder(f)
        f = f.view(B, V, -1, D)
        f = f + self.view_emb.view(1, V, 1, D)

        # pose tokens per view
        pt = torch.stack([self.pose_enc(poses[:, v]) for v in range(V)], 1)  # (B,V,T,D)
        pt = pt + self.view_emb.view(1, V, 1, D)

        mem = torch.cat([f.flatten(1, 2), pt.flatten(1, 2)], 1)
        return mem

    def forward(self, clips, poses):
        B, V, T = clips.shape[:3]
        mem = self._tokens(clips, poses)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        h = self.decoder(q, mem)                        # (B,Q,D)
        Q = h.shape[1]

        logits = self.cls_head(h)
        span = self.span_head(h).sigmoid()               # (B,Q,2) center,width form below
        center, width = span[..., 0], span[..., 1]
        start = (center - width / 2).clamp(0, 1)
        end = (center + width / 2).clamp(0, 1)

        # ---- pose-anchored spatial prediction ----
        hands = poses[:, :, :, HAND_IDX, :]              # (B,V,T,K,3)
        K = len(HAND_IDX)
        a_j = self.anchor_head(h).view(B, Q, V, K)
        a_t = self.anchor_t_head(h).view(B, Q, V, T)
        # invalid (unseen) joints must not be selected
        valid = hands[..., 2] > 0                         # (B,V,T,K)
        j_logit = a_j.unsqueeze(3) + a_t.unsqueeze(4)     # (B,Q,V,T,K)
        j_logit = j_logit.masked_fill(~valid.unsqueeze(1), -1e4)
        w = j_logit.flatten(3).softmax(-1).view(B, Q, V, T, K)
        anchor = torch.einsum("bqvtk,bvtkc->bqvc", w, hands[..., :2])

        off = self.offset_head(h).view(B, Q, V, 2).tanh() * 0.25
        pts = (anchor + off).clamp(0, 1)                  # (B,Q,V,2)

        return {
            "logits": logits,
            "start": start,
            "end": end,
            "points": pts.flatten(2),                     # (B,Q,4)
            "anchor": anchor.flatten(2),
        }
