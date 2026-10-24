"""v1 set criterion + FACT's frame and cross-attention supervision.

Adds, on top of criterion.SetCriterion:
  * frame loss: background/action CE on every sampled frame, for each frame
    branch output (FACT's frame_loss), plus its temporal smoothing loss.
  * f2a loss: a matched query's attention over frames should cover the frames
    of its ground-truth segment (FACT's cross_attn_loss on f2a, softmax over
    frames).
  * a2f loss: each frame inside a segment should attend to the query matched to
    that segment (FACT's cross_attn_loss on a2f, softmax over queries).
  * matching cost: FACT matches tokens to segments mostly by the soft IoU
    between a token's attention and the segment mask; that term is added to
    the v1 geometric cost.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from criterion import SetCriterion, temporal_iou_matrix


def segment_masks(ftimes, spans):
    """(T,), (n,2) -> (n,T) float mask of the sampled frames inside each span."""
    ft = ftimes[None]
    return ((ft >= spans[:, :1] - 1e-6) & (ft < spans[:, 1:] - 1e-6)).float()


class FACTCriterion(SetCriterion):
    def __init__(self, w_frame=1.0, w_f2a=1.0, w_a2f=1.0, w_smooth=0.15,
                 c_attn=1.0, **kw):
        super().__init__(**kw)
        self.w.update(frame=w_frame, f2a=w_f2a, a2f=w_a2f, smooth=w_smooth)
        self.c["attn"] = c_attn

    @torch.no_grad()
    def match(self, out, targets):
        B, Q = out["logits"].shape[:2]
        prob = out["logits"].softmax(-1)
        attn = out["f2a_logit"].softmax(-1) if "f2a_logit" in out else None   # (B,Q,T)
        idxs = []
        for b in range(B):
            t = targets[b]
            if t["labels"].numel() == 0:
                idxs.append((torch.empty(0, dtype=torch.long),
                             torch.empty(0, dtype=torch.long)))
                continue
            dev = out["logits"].device
            gl = t["labels"].to(dev)
            gs, ge = t["spans"][:, 0].to(dev), t["spans"][:, 1].to(dev)
            gp = t["points"].to(dev)
            c_cls = -prob[b][:, gl]
            c_span = torch.cdist(torch.stack([out["start"][b], out["end"][b]], -1),
                                 torch.stack([gs, ge], -1), p=1)
            iou = temporal_iou_matrix(out["start"][b], out["end"][b], gs, ge)
            p = out["points"][b]
            c_pt = torch.cdist(p[:, :2], gp[:, :2]) + torch.cdist(p[:, 2:], gp[:, 2:])
            C = (self.c["cls"] * c_cls + self.c["span"] * c_span
                 + self.c["point"] * c_pt - self.c["iou"] * iou)
            if attn is not None and self.c["attn"] > 0:
                M = segment_masks(t["ftimes"].to(dev), t["spans"].to(dev))       # (n,T)
                a = attn[b]                                                      # (Q,T)
                inter = a @ M.T
                union = torch.minimum(a[:, None] + M[None], torch.ones(())).sum(-1)
                C = C - self.c["attn"] * inter / union.clamp(min=1e-6)
            r, cix = linear_sum_assignment(C.float().cpu().numpy())
            idxs.append((torch.as_tensor(r, dtype=torch.long),
                         torch.as_tensor(cix, dtype=torch.long)))
        self._last_idxs = idxs
        return idxs

    def forward(self, out, targets):
        loss, logs = super().forward(out, targets)      # calls self.match -> caches idxs
        if "frame_logits" not in out:
            return loss, logs
        idxs = self._last_idxs
        dev = out["logits"].device
        flog = out["frame_logits"]                      # (B,nblk,T,2)
        B, nblk, T, _ = flog.shape

        # frame targets: 1 inside any labelled span
        ftgt = torch.zeros(B, T, dtype=torch.long, device=dev)
        for b, t in enumerate(targets):
            if t["labels"].numel():
                M = segment_masks(t["ftimes"].to(dev), t["spans"].to(dev))
                ftgt[b] = (M.sum(0) > 0).long()
        keep_b = torch.tensor([not t.get("rev", False) for t in targets], device=dev)
        if keep_b.any():   # frame boundaries are skipped for time-reversed samples
            l_frame = F.cross_entropy(flog[keep_b].flatten(0, 2),
                                      ftgt[keep_b][:, None].expand(-1, nblk, T).flatten())
        else:
            l_frame = flog.sum() * 0.0
        logp = flog.log_softmax(-1)
        l_smooth = ((logp[:, :, 1:] - logp[:, :, :-1]) ** 2).clamp(max=16).mean()

        # cross-attention supervision on the matched pairs
        l_f2a = torch.zeros((), device=dev)
        l_a2f = torch.zeros((), device=dev)
        n_f2a = n_a2f = 0.0
        for b, (qi, gi) in enumerate(idxs):
            if not qi.numel() or targets[b].get("rev", False):
                continue
            qi, gi = qi.to(dev), gi.to(dev)
            M = segment_masks(targets[b]["ftimes"].to(dev), targets[b]["spans"].to(dev))[gi]  # (m,T)
            keep = M.sum(1) > 0            # a span can miss every sampled frame
            if not keep.any():
                continue
            qi, M = qi[keep], M[keep]
            f2a = out["f2a_logit"][b][qi].log_softmax(-1)            # over frames
            l_f2a = l_f2a - (f2a * M).sum()
            n_f2a += M.sum()
            a2f = out["a2f_logit"][b].log_softmax(-1)[:, qi]          # over queries -> (T,m)
            l_a2f = l_a2f - (a2f * M.T).sum()
            n_a2f += M.sum()
        l_f2a = l_f2a / max(n_f2a, 1.0)
        l_a2f = l_a2f / max(n_a2f, 1.0)

        loss = (loss + self.w["frame"] * l_frame + self.w["smooth"] * l_smooth
                + self.w["f2a"] * l_f2a + self.w["a2f"] * l_a2f)
        logs.update(frame=l_frame.detach(), f2a=l_f2a.detach(), a2f=l_a2f.detach(),
                    smooth=l_smooth.detach(), total=loss.detach())
        return loss, logs
