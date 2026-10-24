"""Multi-view pose DETR + FACT-style frame branch (Lu & Elhamifar, CVPR 2024).

FACT models a video at two levels in parallel - frames and action tokens - and
lets them exchange information through cross-attention, supervising the
attention maps themselves. Our DETR queries already play the role of FACT's
action tokens, so this adds the missing half:

  * a frame branch: per-frame features (video + pose, both views) refined by a
    dilated temporal conv stack, classified background / action per frame.
    Temporal labels are frame indices / n_frames (97% of start/end sit exactly
    on that grid), so per-frame targets lose nothing.
  * one FACT update block: queries attend to frames (f2a), then frames attend
    back to the queries (a2f), then the frame branch runs again.
  * both attention maps are returned as logits so the criterion can supervise
    them against the matched ground-truth segment.

Everything spatial (pose anchor + offset) is unchanged from model.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MultiViewPoseDETR, HAND_IDX, mlp


class DilatedResidual(nn.Module):
    def __init__(self, dim, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B,D,T)
        return x + self.drop(self.proj(F.relu(self.conv(x))))


class FrameBranch(nn.Module):
    """MS-TCN-style stage, sized for 16 frames (receptive field 31 >= T)."""

    def __init__(self, dim, dilations=(1, 2, 4, 8), n_out=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(DilatedResidual(dim, d, dropout) for d in dilations)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, n_out)

    def forward(self, x):                      # (B,T,D)
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x.transpose(1, 2))
        return x, self.cls(x)


class CrossAttention(nn.Module):
    """Single-head cross-attention that exposes its logits (FACT's X2Y_map)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.drop = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, y, x, y_pos=None, x_pos=None):
        """y (B,Ny,D) attends to x (B,Nx,D). Returns updated y and logits (B,Ny,Nx)."""
        qy = self.norm_q(y) + (y_pos if y_pos is not None else 0)
        kx = self.norm_kv(x)
        logit = (self.q(qy) @ self.k(kx + (x_pos if x_pos is not None else 0)).transpose(1, 2)) * self.scale
        y = y + self.drop(self.o(logit.softmax(-1) @ self.v(kx)))
        y = y + self.drop(self.ffn(y))
        return y, logit


class MultiViewPoseDETRFACT(MultiViewPoseDETR):
    def __init__(self, backbone, feat_dim, dim=256, num_queries=10, num_classes=3,
                 num_frames=16, tokens_per_frame=49, dec_layers=4, enc_layers=2):
        super().__init__(backbone, feat_dim, dim, num_queries, num_classes,
                         num_frames, tokens_per_frame, dec_layers, enc_layers)
        self.frame_in = nn.Linear(2 * dim, dim)          # fuse the two views per frame
        self.frame_pos = nn.Parameter(torch.randn(num_frames, dim) * 0.02)
        self.fb1 = FrameBranch(dim)                        # input block frame branch
        self.f2a = CrossAttention(dim)                     # queries <- frames
        self.a2f = CrossAttention(dim)                     # frames  <- queries
        self.fb2 = FrameBranch(dim)                        # update block frame branch

    def _tokens_fact(self, clips, poses):
        """Same memory as model.py, plus per-frame features for the frame branch."""
        B, V, T = clips.shape[:3]
        feats = self.backbone(clips.flatten(0, 1))         # (B*V, Tf, S, C)
        f = self.proj(feats)
        BV, Tf, S, D = f.shape
        g = self.grid
        pos = (self.pos_t[:Tf].view(Tf, 1, D)
               + (self.pos_h.view(g, 1, D) + self.pos_w.view(1, g, D)).view(1, g * g, D)[:, :S])
        f = self.encoder((f + pos).flatten(1, 2))          # (B*V, Tf*S, D)
        f = f.view(B, V, Tf * S, D) + self.view_emb.view(1, V, 1, D)

        pt = torch.stack([self.pose_enc(poses[:, v]) for v in range(V)], 1)   # (B,V,T,D)
        pt = pt + self.view_emb.view(1, V, 1, D)
        mem = torch.cat([f.flatten(1, 2), pt.flatten(1, 2)], 1)

        # per-frame video feature: pool space, then bring the backbone's Tf slots
        # (Swin3D halves time) back to the pose rate T
        vid = f.view(B * V, Tf, S, D).mean(2).transpose(1, 2)                 # (B*V,D,Tf)
        vid = F.interpolate(vid, size=T, mode="linear", align_corners=False)
        vid = vid.transpose(1, 2).reshape(B, V, T, D)
        per_view = vid + pt                                                    # (B,V,T,D)
        frames = self.frame_in(torch.cat([per_view[:, 0], per_view[:, 1]], -1))
        return mem, frames

    def forward(self, clips, poses):
        B, V, T = clips.shape[:3]
        mem, frames = self._tokens_fact(clips, poses)
        fpos = self.frame_pos[:T].unsqueeze(0)

        # input block: frame branch on its own, DETR decoder on the full memory
        fr1, flog1 = self.fb1(frames + fpos)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        h = self.decoder(q, mem)                                               # (B,Q,D)

        # update block: bidirectional frame <-> action exchange
        h, f2a_logit = self.f2a(h, fr1, x_pos=fpos)                            # (B,Q,T)
        fr2, a2f_logit = self.a2f(fr1, h, y_pos=fpos)                          # (B,T,Q)
        _, flog2 = self.fb2(fr2)

        out = self._heads(h, poses, B, V, T)
        out.update({
            "frame_logits": torch.stack([flog1, flog2], 1),                   # (B,2,T,2)
            "f2a_logit": f2a_logit,
            "a2f_logit": a2f_logit,
        })
        return out

    def _heads(self, h, poses, B, V, T):
        """Prediction heads, identical to model.py."""
        Q = h.shape[1]
        logits = self.cls_head(h)
        span = self.span_head(h).sigmoid()
        center, width = span[..., 0], span[..., 1]
        start = (center - width / 2).clamp(0, 1)
        end = (center + width / 2).clamp(0, 1)

        hands = poses[:, :, :, HAND_IDX, :]                                    # (B,V,T,K,3)
        K = len(HAND_IDX)
        a_j = self.anchor_head(h).view(B, Q, V, K)
        a_t = self.anchor_t_head(h).view(B, Q, V, T)
        valid = hands[..., 2] > 0
        j_logit = a_j.unsqueeze(3) + a_t.unsqueeze(4)
        j_logit = j_logit.masked_fill(~valid.unsqueeze(1), -1e4)
        w = j_logit.flatten(3).softmax(-1).view(B, Q, V, T, K)
        anchor = torch.einsum("bqvtk,bvtkc->bqvc", w, hands[..., :2])
        off = self.offset_head(h).view(B, Q, V, 2).tanh() * 0.25
        pts = (anchor + off).clamp(0, 1)
        # kept for inference-time hedging: which (frame, hand joint) each query picked
        self._anchor_w, self._hands, self._offset = w, hands[..., :2], off
        return {"logits": logits, "start": start, "end": end,
                "points": pts.flatten(2), "anchor": anchor.flatten(2)}
